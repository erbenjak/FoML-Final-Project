import events as e
import numpy as np
import random
import time

from ..rl_models.BaseQLearningModelGradient import BaseQLearningModel


class GradientModel(BaseQLearningModel):
    REWARDS = {
        e.INVALID_ACTION: -300,
        ###############
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.WAITED: -50,
        ###############
        e.SURVIVED_ROUND: 100,
        e.KILLED_SELF: -3000,
        e.COIN_COLLECTED: 300,
        e.CRATE_DESTROYED: 200,
        e.KILLED_OPPONENT: 2000,
        e.GOT_KILLED: -700,
        e.COIN_FOUND: 50,
        ################ DISABLED EVENTS ###########################
        e.BOMB_DROPPED: 150,
        e.BOMB_EXPLODED: 0,
        e.OPPONENT_ELIMINATED: 500,
        ############### CUSTOM EVENTS ##############################
        'DEC_COIN_DIST': 100,
        "INC_COIN_DIST": -100,
        'DEC_CRATE_DIST': 80,
        "INC_CRATE_DIST": -80,
        'DEC_BOMB_DIST': -100,
        "INC_BOMB_DIST": 100,
        "KEEP_BOMB_DIST": -800,
        "GOOD_BOMB": 600,
        "VERY_GOOD_BOMB": 800,
        "BAD_BOMB": -800,
        "SURVIVED_BOMB": 100,
        "BOMB_AND_ESCAPE": 800,
        "BOMB_NO_ESCAPE": -1400,
        "GOOD_EVASION": 300,
        "VERY_GOOD_EVASION": 500
    }

    """
    This is an implementation of the BaseQLearningModel.
    REMINDER: x- and y-axis of the board point right and down (matrix-view)
                they do NOT point right and up (typical physical view)
    """

    def __init__(self, logger, max_feature_size, path_q_table, path_seen_representations):
        BaseQLearningModel.__init__(self, logger, max_feature_size, path_q_table, path_seen_representations)

    ############### Necessary Overwrites ##############################################
    def compute_additional_rewards(self, events, new_state, old_state, action, short_memory):
        """
        Takes a set of events produced by the game engine and adds some custom events to be able to
        add some additional self-defined 'custom' events
        """
        events = self.addSurrivedABombEvent(events, old_state)
        events = self.addCoinDistanceEvents(events, new_state, old_state)
        events = self.addBombDistanceEvents(events, new_state, old_state)
        events = self.addCrateDistanceEvents(events, new_state, old_state)
        events = self.placedBombEvent(events, new_state, old_state)
        events = self.evasion_event(events, new_state, old_state)
        # events = self.add_escapable_bomb_event(events, old_state, new_state, action, short_memory)
        return events

    def evasion_event(self, events, new_state, old_state):

        pos_agent = (old_state['self'])[3]
        bombs = old_state['bombs']
        field = old_state['field']
        explosion_map = old_state["explosion_map"]
        if bombs == None:
            return events

        bombPos = self.bomb_direction_feature(pos_agent, bombs)

        if bombPos[0] != 0 or bombPos[1] != 0:
            # print("not on bomb")
            return events
        densities = self.calc_density(pos_agent, field, explosion_map)
        bombs_in_new = new_state['bombs']
        pos_agent_new = (new_state['self'])[3]

        if bombs == None:
            return events

        new_bombPos = self.bomb_direction_feature(pos_agent_new, bombs_in_new)

        '''
        new_bombPos[0] = -1 -> bomb lies east
        new_bombPos[0] = 1 -> bomb lies west
        new_bombPos[1] = -1 -> bomb lies south
        new_bombPos[1] = 1 -> bomb lies north

        densities[0] = 0 -> W
        densities[0] = 1 -> E
        densities[0] = 2 -> S
        densities[0] = 3 -> N

        densities[1] = 0 -> NW
        densities[1] = 1 -> NE
        densities[1] = 2 -> SW
        densities[1] = 3 -> SE

        '''
        if new_bombPos[0] == -1 and (densities[0] == 0 and (densities[1] == 0 or densities[1] == 2)):
            events.append("VERY_GOOD_EVASION")
            return events
        if new_bombPos[0] == 1 and (densities[0] == 1 and (densities[1] == 1 or densities[1] == 3)):
            events.append("VERY_GOOD_EVASION")
            return events
        if new_bombPos[1] == -1 and (densities[0] == 3 and (densities[1] == 0 or densities[1] == 1)):
            events.append("VERY_GOOD_EVASION")
            return events
        if new_bombPos[1] == 1 and (densities[0] == 2 and (densities[1] == 2 or densities[1] == 3)):
            events.append("VERY_GOOD_EVASION")
            return events
        if new_bombPos[0] == -1 and (densities[0] == 0 or densities[1] == 0 or densities[1] == 2):
            events.append("GOOD_EVASION")
            return events
        if new_bombPos[0] == 1 and (densities[0] == 1 or densities[1] == 1 or densities[1] == 3):
            events.append("GOOD_EVASION")
            return events
        if new_bombPos[1] == -1 and (densities[0] == 3 or densities[1] == 0 or densities[1] == 1):
            events.append("GOOD_EVASION")
            return events
        if new_bombPos[1] == 1 and (densities[0] == 2 or densities[1] == 2 or densities[1] == 3):
            events.append("GOOD_EVASION")
            return events

        return events

    def placedBombEvent(self, events, new_state, old_state):
        agentPos = new_state['self'][3]
        field = new_state['field']

        bomb_positions = []
        new_bomb = False

        for bomb in new_state['bombs']:
            if bomb[1] == 3:
                new_bomb = True
                bomb_positions.append(bomb[0])

        if new_bomb is False:
            return events

        own_bomb = False
        for bomb in bomb_positions:
            if bomb[0] == agentPos[0] and bomb[1] == agentPos[1]:
                own_bomb = True

        if own_bomb == False:
            return events

        # check if a bomb was placed directly besides an opponent
        for opponent in old_state['others']:
            opponent_position = opponent[3]

            if agentPos[0] + 1 == opponent_position[0] and agentPos[1] == opponent_position[1]:
                events.append("VERY_GOOD_BOMB")
                return events

            if agentPos[0] - 1 == opponent_position[0] and agentPos[1] == opponent_position[1]:
                events.append("VERY_GOOD_BOMB")
                return events

            if agentPos[0] == opponent_position[0] and agentPos[1] + 1 == opponent_position[1]:
                events.append("VERY_GOOD_BOMB")
                return events

            if agentPos[0] == opponent_position[0] and agentPos[1] - 1 == opponent_position[1]:
                events.append("VERY_GOOD_BOMB")
                return events

        if field[agentPos[0] + 1][agentPos[1]] == 1:
            events.append("GOOD_BOMB")
            return events

        if field[agentPos[0] - 1][agentPos[1]] == 1:
            events.append("GOOD_BOMB")
            return events

        if field[agentPos[0]][agentPos[1] + 1] == 1:
            events.append("GOOD_BOMB")
            return events

        if field[agentPos[0]][agentPos[1] - 1] == 1:
            events.append("GOOD_BOMB")
            return events

        events.append("BAD_BOMB")
        return events

    def add_escapable_bomb_event(self, events, old_state, new_state, action, memory):
        """
        The idea is to punish any move that would lead to one no loger being able to escape a bomb
        """
        # at least 2 moves need to be played
        if len(memory) < 2:
            return events

        memory_copy = memory.copy()
        # self.logger.info("")
        lastMove = memory_copy.popleft()
        secondLastMove = memory_copy.popleft()

        if secondLastMove[1] != "BOMB":
            return events

        if lastMove[1] == "WAIT" or e.INVALID_ACTION in events:
            events.append("BOMB_NO_ESCAPE")
            return events

        new_agent_position = new_state["self"][3]
        new_field = new_state["field"]

        # now one needs to check if the agent can still escape from the newly reached position
        if lastMove == "UP" or lastMove == "DOWN":
            if new_agent_position[1] % 2 == 1:
                if new_field[new_agent_position[0] - 1][new_agent_position[1]] == 0:
                    events.append("BOMB_AND_ESCAPE")
                    return events
                if new_field[new_agent_position[0] + 1][new_agent_position[1]] == 0:
                    events.append("BOMB_AND_ESCAPE")
                    return events

            if new_field[new_agent_position[0]][new_agent_position[1] - 1] != 0 or \
                    new_field[new_agent_position[0]][new_agent_position[1] + 1] != 0:
                events.append("BOMB_NO_ESCAPE")
                return events

        # now one needs to check if the agent can still escape from the newly reached position
        if lastMove == "LEFT" or lastMove == "RIGHT":
            if new_agent_position[0] % 2 == 1:
                if new_field[new_agent_position[0]][new_agent_position[1] - 1] == 0:
                    events.append("BOMB_AND_ESCAPE")
                    return events
                if new_field[new_agent_position[0]][new_agent_position[1] + 1] == 0:
                    events.append("BOMB_AND_ESCAPE")
                    return events

            if new_field[new_agent_position[0] - 1][new_agent_position[1]] != 0 or \
                    new_field[new_agent_position[0] + 1][new_agent_position[1]] != 0:
                events.append("BOMB_NO_ESCAPE")
                return events

        return events

    def addSurrivedABombEvent(self, events, old_state):
        if e.GOT_KILLED in events or e.KILLED_SELF in events:
            return events

        for bomb in old_state['bombs']:
            countdown = bomb[1]
            if countdown == 0:
                events.append("SURVIVED_BOMB")
                return events
        return events

    def addBombDistanceEvents(self, events, new_state, old_state):
        if len(old_state['bombs']) == 0:
            return events

        if len(new_state['bombs']) == 0:
            # bomb has exploded therefore other events apply
            return events

        old_distance = self.calc_bomb_dist(old_state['self'][3], old_state['bombs'])
        new_distance = self.calc_bomb_dist(new_state['self'][3], new_state['bombs'])

        # self.logger.info("old_distance: "+  str(old_distance))
        # self.logger.info("new_distance: "+  str(new_distance))

        if old_distance < new_distance:
            if old_distance > 4:
                return events
            events.append("INC_BOMB_DIST")
            return events

        if new_distance < old_distance:
            if new_distance > 4:
                return events
            events.append("DEC_BOMB_DIST")
            return events

        events.append("KEEP_BOMB_DIST")
        return events

    def addCoinDistanceEvents(self, events, new_state, old_state):
        # if a coin was collected the distance was shortened:
        if (e.COIN_COLLECTED in events):
            return events

        # since no coin was present the agent could not possibly moce towards it
        if len(old_state['coins']) == 0:
            return events

        old_distance = self.calc_coin_dist(old_state['self'][3], old_state['coins'])
        new_distance = self.calc_coin_dist(new_state['self'][3], new_state['coins'])
        if old_distance < new_distance:
            events.append("INC_COIN_DIST")
            return events

        if new_distance < old_distance:
            events.append("DEC_COIN_DIST")
            return events

        return events

    def addCrateDistanceEvents(self, events, new_state, old_state):
        crates_old = np.argwhere(old_state['field'] == 1)
        crates_new = np.argwhere(new_state['field'] == 1)
        # print("crates old position:" + str(crates_old))
        if len(crates_old) == 0:
            return events

        old_distance = self.calc_crate_dist(old_state['self'][3], crates_old)
        new_distance = self.calc_crate_dist(new_state['self'][3], crates_new)

        if old_distance < new_distance:
            events.append("INC_CRATE_DIST")
            return events

        if new_distance < old_distance:
            events.append("DEC_CRATE_DIST")
            return events

        return events

    @staticmethod
    def calc_crate_dist(agent, crates):
        if (len(crates) == 0):
            return 0
        overall_dist = 1000

        for crate in crates:

            x_dist = abs(agent[0] - crate[0])
            y_dist = abs(agent[1] - crate[1])
            total_dist = x_dist + y_dist
            # one now needs to account for blocks in the way of the coin

            if x_dist == 0 and (agent[0] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if y_dist == 0 and (agent[1] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if (total_dist < overall_dist):
                overall_dist = total_dist
        # print("crate dist:" + str(overall_dist))
        return overall_dist

    @staticmethod
    def calc_bomb_dist(agent, bombs):
        if (len(bombs) == 0):
            return 0
        overall_dist = 1000
        for bomb in bombs:

            x_dist = abs(agent[0] - bomb[0][0])
            y_dist = abs(agent[1] - bomb[0][1])
            total_dist = x_dist + y_dist
            # one now needs to account for blocks in the way of the coin

            if x_dist == 0 and (agent[0] % 2) == 0 and y_dist != 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if y_dist == 0 and (agent[1] % 2) == 0 and x_dist != 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if (total_dist < overall_dist):
                overall_dist = total_dist
        # print("Bomb distance:" + str(overall_dist))
        return overall_dist

    @staticmethod
    def calc_coin_dist(agent, coins):
        if (len(coins) == 0):
            return 0
        overall_dist = 1000

        for coin in coins:

            x_dist = abs(agent[0] - coin[0])
            y_dist = abs(agent[1] - coin[1])
            total_dist = x_dist + y_dist
            # one now needs to account for blocks in the way of the coin
            if x_dist == 0 and (agent[0] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if y_dist == 0 and (agent[1] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if (total_dist < overall_dist):
                overall_dist = total_dist
        return overall_dist

    def calculateReward(self, events):
        """
        Calculate the reward a agent receives for a certain events.
        The return may be any float.
        """
        reward = 0
        # self.logger.info(str(events))
        # print(events)
        for event in events:
            reward += self.REWARDS[event]
        self.logger.info(events)
        # print("reward:" + str(reward))
        # time.sleep(10)
        return reward

    def calculateFeaturesFromState(self, state):
        """
        Takes a gamestate and extracts the features which are required to take good decisions.
        """
        encoded_state = np.array([])

        # some minor data prep for the calculating methods
        pos_agent = (state['self'])[3]
        coins = state['coins']
        field = state['field']
        explosion_map = state['explosion_map']
        bombs = state['bombs']

        opponent_positions = []
        for opponent in state['others']:
            opponent_positions.append(opponent[3])

        # adds two features
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_coin(pos_agent, coins, field))
        # adds eight feature
        encoded_state = np.append(encoded_state, self.calc_surrounding_features(pos_agent, field, explosion_map, bombs,
                                                                                opponent_positions))
        # adds one feature
        encoded_state = np.append(encoded_state,
                                  self.bombing_effective_feature(state['self'], field, opponent_positions))
        # adds three features
        encoded_state = np.append(encoded_state, self.bomb_direction_feature(pos_agent, bombs))
        # adds two features
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_crate(pos_agent, field))
        # adds two features
        encoded_state = np.append(encoded_state,
                                  self.calc_distance_and_direction_opponent(pos_agent, field, opponent_positions))
        # adds two features
        encoded_state = np.append(encoded_state, self.calc_density(pos_agent, field, explosion_map))
        # print(encoded_state[2:6])
        return encoded_state

    def calc_density(self, pos_agent, field, explosion_map):

        line_grad = self.calc_direction_density(pos_agent, field, explosion_map)
        quarter_grad = self.calc_quarter_density(pos_agent, field, explosion_map)

        feature = np.array([line_grad, quarter_grad])
        return feature

    def calc_quarter_density(self, pos_agent, field, explosion_map):
        '''
        789----
        634----
        521----
        ---A---       <- # = relevant tiles for north-west-quarter density
        -------
        -------
        '''

        x = pos_agent[0]
        y = pos_agent[1]

        '''
        north-west
        '''

        tiles = np.zeros((9))

        tiles[0] = field[x - 1][y - 1]

        if y - 2 < 0:
            tiles[1] = -1
        else:
            tiles[1] = field[x - 1][y - 2]

        if x - 2 < 0 or y - 2 < 0:
            tiles[2] = -1
        else:
            tiles[2] = field[x - 2][y - 2]

        if x - 2 < 0:
            tiles[3] = -1
        else:
            tiles[3] = field[x - 2][y - 1]

        if y - 3 < 0:
            tiles[4] = -1
        else:
            tiles[4] = field[x - 1][y - 3]

        if y - 3 < 0 or x - 2 < 0:
            tiles[5] = -1
        else:
            tiles[5] = field[x - 2][y - 3]

        if y - 3 < 0 or x - 3 < 0:
            tiles[6] = -1
        else:
            tiles[6] = field[x - 3][y - 3]

        if y - 2 < 0 or x - 3 < 0:
            tiles[7] = -1
        else:
            tiles[7] = field[x - 3][y - 2]

        if x - 3 < 0:
            tiles[8] = -1
        else:
            tiles[8] = field[x - 3][y - 1]

        free_tiles = len(np.where(tiles == 0)[0])
        density_north_west = free_tiles / 9

        '''
        south-west
        '''

        tiles = np.zeros((9))

        tiles[0] = field[x - 1][y + 1]

        if x - 2 < 0:
            tiles[1] = -1
        else:
            tiles[1] = field[x - 2][y + 1]

        if x - 2 < 0 or y + 2 > 16:
            tiles[2] = -1
        else:
            tiles[2] = field[x - 2][y + 2]

        if y + 2 > 16:
            tiles[3] = -1
        else:
            tiles[3] = field[x - 1][y + 2]

        if x - 3 < 0:
            tiles[4] = -1
        else:
            tiles[4] = field[x - 3][y + 1]

        if x - 3 < 0 or y + 2 > 16:
            tiles[5] = -1
        else:
            tiles[5] = field[x - 3][y + 2]

        if x - 3 < 0 or y + 3 > 16:
            tiles[6] = -1
        else:
            tiles[6] = field[x - 3][y + 3]

        if x - 2 < 0 or y + 3 > 16:
            tiles[7] = -1
        else:
            tiles[7] = field[x - 2][y + 3]

        if y + 3 > 16:
            tiles[8] = -1
        else:
            tiles[8] = field[x - 1][y + 3]

        free_tiles = len(np.where(tiles == 0)[0])
        density_south_west = free_tiles / 9

        '''
        north-east
        '''

        tiles = np.zeros((9))

        tiles[0] = field[x + 1][y - 1]

        if y - 2 < 0:
            tiles[1] = -1
        else:
            tiles[1] = field[x + 1][y - 2]

        if x + 2 > 16 or y - 2 < 0:
            tiles[2] = -1
        else:
            tiles[2] = field[x + 2][y - 2]

        if x + 2 > 16:
            tiles[3] = -1
        else:
            tiles[3] = field[x + 2][y - 1]

        if y - 3 < 0:
            tiles[4] = -1
        else:
            tiles[4] = field[x + 1][y - 3]

        if y - 3 < 0 or x + 2 > 16:
            tiles[5] = -1
        else:
            tiles[5] = field[x + 2][y - 3]

        if y - 3 < 0 or x + 3 > 16:
            tiles[6] = -1
        else:
            tiles[6] = field[x + 3][y - 3]

        if y - 2 < 0 or x + 3 > 16:
            tiles[7] = -1
        else:
            tiles[7] = field[x + 3][y - 2]

        if x + 3 > 16:
            tiles[8] = -1
        else:
            tiles[8] = field[x + 3][y - 1]

        free_tiles = len(np.where(tiles == 0)[0])
        density_north_east = free_tiles / 9

        '''
        south-east
        '''

        tiles = np.zeros((9))

        tiles[0] = field[x + 1][y + 1]

        if x + 2 > 16:
            tiles[1] = -1
        else:
            tiles[1] = field[x + 2][y + 1]

        if x + 2 > 16 or y + 2 > 16:
            tiles[2] = -1
        else:
            tiles[2] = field[x + 2][y + 2]

        if y + 2 > 16:
            tiles[3] = -1
        else:
            tiles[3] = field[x + 1][y + 2]

        if x + 3 > 16:
            tiles[4] = -1
        else:
            tiles[4] = field[x + 3][y + 1]

        if x + 3 > 16 or y + 2 > 16:
            tiles[5] = -1
        else:
            tiles[5] = field[x + 3][y + 2]

        if x + 3 > 16 or y + 3 > 16:
            tiles[6] = -1
        else:
            tiles[6] = field[x + 3][y + 3]

        if x + 2 > 16 or y + 3 > 16:
            tiles[7] = -1
        else:
            tiles[7] = field[x + 2][y + 3]

        if y + 3 > 16:
            tiles[8] = -1
        else:
            tiles[8] = field[x + 1][y + 3]

        free_tiles = len(np.where(tiles == 0)[0])
        density_south_east = free_tiles / 9

        densities = np.array([density_north_west, density_north_east, density_south_west, density_south_east])
        """
        feature[0] = left
        feature[1] = right
        feature[2] = down
        feature[3] = up
        """
        # self.logger.info("started with the following agent position: "+str(pos_agent))
        feature = np.zeros((4))

        feature[0] = field[x - 1][y]
        feature[1] = field[x + 1][y]
        feature[2] = field[x][y + 1]
        feature[3] = field[x][y - 1]

        explosions = np.zeros((4))

        explosions[0] = explosion_map[x - 1][y]
        explosions[1] = explosion_map[x + 1][y]
        explosions[2] = explosion_map[x][y + 1]
        explosions[3] = explosion_map[x][y - 1]

        for i in np.arange(len(feature)):
            if int(feature[i]) != -1 and int(explosions[i]) != 0:
                feature[i] = 2

        if int(feature[0]) != 0 and int(feature[3]) != 0:
            densities[0] = -1
        if int(feature[1]) != 0 and int(feature[3]) != 0:
            densities[1] = -1
        if int(feature[0]) != 0 and int(feature[2]) != 0:
            densities[2] = -1
        if int(feature[1]) != 0 and int(feature[2]) != 0:
            densities[3] = -1
        density_grad = np.argmax(densities)
        # print("diagnola density grad:" + str(densities) + str(density_grad))
        return density_grad

    def calc_direction_density(self, pos_agent, field, explosion_map):
        '''
        calculating the density of free tiles in north, south, west, and east direction
        9ABCDEF
        -45678-
        --123--
        ---A---      <- # = relevant tiles for northern density
        -------
        -------
        '''
        x = pos_agent[0]
        y = pos_agent[1]

        '''
        western density
        '''
        range = 2
        tiles = np.zeros((15))

        tiles[0] = field[x - 1][y - 1]
        tiles[1] = field[x - 1][y]
        tiles[2] = field[x - 1][y + 1]
        if x - 2 < 0:
            tiles[3:8] = -1
        else:
            if y - 2 < 0:
                tiles[3] = -1
            else:
                tiles[3] = field[x - 2][y - 2]
            tiles[4] = field[x - 2][y - 1]
            tiles[5] = field[x - 2][y]
            tiles[6] = field[x - 2][y + 1]
            if y + 2 > 16:
                tiles[7] = -1
            else:
                tiles[7] = field[x - 2][y + 2]
        if x - 3 < 0:
            tiles[8:15] = -1
        else:
            if y - 3 < 0:
                tiles[8] = -1
            else:
                tiles[8] = field[x - 3][y - 3]
            if y - 2 < 0:
                tiles[9] = -1
            else:
                tiles[9] = field[x - 3][y - 2]
            tiles[10] = field[x - 3][y - 1]
            tiles[11] = field[x - 3][y]
            tiles[12] = field[x - 3][y + 1]
            if y + 2 > 16:
                tiles[13] = -1
            else:
                tiles[13] = field[x - 3][y + 2]
            if y + 3 > 16:
                tiles[14] = -1
            else:
                tiles[14] = field[x - 3][y + 3]
        free_tiles = len(np.where(tiles == 0)[0])
        density_west = free_tiles / 15
        '''
        southern density
        '''
        tiles = np.zeros((15))

        tiles[0] = field[x - 1][y + 1]
        tiles[1] = field[x][y + 1]
        tiles[2] = field[x + 1][y + 1]
        if y + 2 > 16:
            tiles[3:8] = -1
        else:
            if x - 2 < 0:
                tiles[3] = -1
            else:
                tiles[3] = field[x - 2][y + 2]
            tiles[4] = field[x - 1][y + 2]
            tiles[5] = field[x][y + 2]
            tiles[6] = field[x + 1][y + 2]
            if x + 2 > 16:
                tiles[7] = -1
            else:
                tiles[7] = field[x + 2][y + 2]
        if y + 3 > 16:
            tiles[8:15] = -1
        else:
            if x - 3 < 0:
                tiles[8] = -1
            else:
                tiles[8] = field[x - 3][y + 3]
            if x - 2 < 0:
                tiles[9] = -1
            else:
                tiles[9] = field[x - 2][y + 3]
            tiles[10] = field[x - 1][y + 3]
            tiles[11] = field[x][y + 3]
            tiles[12] = field[x + 1][y + 3]
            if x + 2 > 16:
                tiles[13] = -1
            else:
                tiles[13] = field[x + 2][y + 3]
            if x + 3 > 16:
                tiles[14] = -1
            else:
                tiles[14] = field[x + 3][y + 3]
        free_tiles = len(np.where(tiles == 0)[0])
        density_south = free_tiles / 15
        '''
        eastern density
        '''

        tiles = np.zeros((15))

        tiles[0] = field[x + 1][y - 1]
        tiles[1] = field[x + 1][y]
        tiles[2] = field[x + 1][y + 1]
        if x + 2 > 16:
            tiles[3:8] = -1
        else:
            if y - 2 < 0:
                tiles[3] = -1
            else:
                tiles[3] = field[x + 2][y - 2]
            tiles[4] = field[x + 2][y - 1]
            tiles[5] = field[x + 2][y]
            tiles[6] = field[x + 2][y + 1]
            if y + 2 > 16:
                tiles[7] = -1
            else:
                tiles[7] = field[x + 2][y + 2]
        if x + 3 > 16:
            tiles[8:15] = -1
        else:
            if y - 3 < 0:
                tiles[8] = -1
            else:
                tiles[8] = field[x + 3][y - 3]
            if y - 2 < 0:
                tiles[9] = -1
            else:
                tiles[9] = field[x + 3][y - 2]
            tiles[10] = field[x + 3][y - 1]
            tiles[11] = field[x + 3][y]
            tiles[12] = field[x + 3][y + 1]
            if y + 2 > 16:
                tiles[13] = -1
            else:
                tiles[13] = field[x + 3][y + 2]
            if y + 3 > 16:
                tiles[14] = -1
            else:
                tiles[14] = field[x + 3][y + 3]
        free_tiles = len(np.where(tiles == 0)[0])
        density_east = free_tiles / 15

        '''
        northern density
        '''
        tiles = np.zeros((15))

        tiles[0] = field[x - 1][y - 1]
        tiles[1] = field[x][y - 1]
        tiles[2] = field[x + 1][y - 1]
        if y - 2 < 0:
            tiles[3:8] = -1
        else:
            if x - 2 < 0:
                tiles[3] = -1
            else:
                tiles[3] = field[x - 2][y - 2]
            tiles[4] = field[x - 1][y - 2]
            tiles[5] = field[x][y - 2]
            tiles[6] = field[x + 1][y - 2]
            if x + 2 > 16:
                tiles[7] = -1
            else:
                tiles[7] = field[x + 2][y - 2]
        if y - 3 < 0:
            tiles[8:15] = -1
        else:
            if x - 3 < 0:
                tiles[8] = -1
            else:
                tiles[8] = field[x - 3][y - 3]
            if x - 2 < 0:
                tiles[9] = -1
            else:
                tiles[9] = field[x - 2][y - 3]
            tiles[10] = field[x - 1][y - 3]
            tiles[11] = field[x][y - 3]
            tiles[12] = field[x + 1][y - 3]
            if x + 2 > 16:
                tiles[13] = -1
            else:
                tiles[13] = field[x + 2][y - 3]
            if x + 3 > 16:
                tiles[14] = -1
            else:
                tiles[14] = field[x + 3][y - 3]
        free_tiles = len(np.where(tiles == 0)[0])
        density_north = free_tiles / 15

        densities = np.array([density_west, density_east, density_south, density_north])
        """
        feature[0] = left
        feature[1] = right
        feature[2] = down
        feature[3] = up
        """
        # self.logger.info("started with the following agent position: "+str(pos_agent))
        feature = np.zeros((4))

        feature[0] = field[x - 1][y]
        feature[1] = field[x + 1][y]
        feature[2] = field[x][y + 1]
        feature[3] = field[x][y - 1]

        explosions = np.zeros((4))

        explosions[0] = explosion_map[x - 1][y]
        explosions[1] = explosion_map[x + 1][y]
        explosions[2] = explosion_map[x][y + 1]
        explosions[3] = explosion_map[x][y - 1]

        for i in np.arange(len(feature)):
            if int(feature[i]) != -1 and int(explosions[i]) != 0:
                feature[i] = 2

        for i in np.arange(len(feature)):
            if int(feature[i]) != 0:
                densities[i] = -1
        density_grad = np.argmax(densities)
        # print("line density grad:" + str(densities) + str(density_grad))
        return density_grad

    def calc_distance_and_direction_coin(self, pos_agent, coins, field):
        '''
        Simple encoding -1, 0, 1 for Left, no_x, Right
                        -1, 0, 1 for Up, no_y, Down
        '''
        feature = np.zeros((2))
        overall_dist = 1000
        best_coin = None

        if len(coins) == 0:
            return feature

        for coin in coins:
            x_dist = abs(pos_agent[0] - coin[0])
            y_dist = abs(pos_agent[1] - coin[1])
            total_dist = x_dist + y_dist

            # one now needs to account for blocks in the way of the coin
            if x_dist == 0 and (pos_agent[0] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if y_dist == 0 and (pos_agent[1] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if (total_dist < overall_dist):
                overall_dist = total_dist
                best_coin = coin

        if pos_agent[0] < best_coin[0]:
            feature[0] = -1
        elif pos_agent[0] > best_coin[0]:
            feature[0] = 1

        if pos_agent[1] < best_coin[1]:
            feature[1] = -1
        elif pos_agent[1] > best_coin[1]:
            feature[1] = 1

        return feature

    def calc_distance_and_direction_crate(self, pos_agent, field):
        '''
        Simple encoding -1, 0, 1 for Left, no_x, Right
                        -1, 0, 1 for Up, no_y, Down
        '''
        feature = np.zeros((2))
        overall_dist = 1000
        best_crate = None

        crates = np.argwhere(field == 1)

        if len(crates) == 0:
            # self.logger.info("No more crates left to destroy ")
            return feature

        for crate in crates:
            x_dist = abs(pos_agent[0] - crate[0])
            y_dist = abs(pos_agent[1] - crate[1])
            total_dist = x_dist + y_dist

            # one now needs to account for blocks in the way of the coin
            if x_dist == 0 and (pos_agent[0] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if y_dist == 0 and (pos_agent[1] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if (total_dist < overall_dist):
                overall_dist = total_dist
                best_crate = crate

        if pos_agent[0] < best_crate[0]:
            feature[0] = -1
        elif pos_agent[0] > best_crate[0]:
            feature[0] = 1

        if pos_agent[1] < best_crate[1]:
            feature[1] = -1
        elif pos_agent[1] > best_crate[1]:
            feature[1] = 1

        return feature

    def calc_distance_and_direction_opponent(self, pos_agent, field, opponents_pos):
        '''
        Simple encoding -1, 0, 1 for Left, no_x, Right
                        -1, 0, 1 for Up, no_y, Down
        '''
        feature = np.zeros((2))

        if len(opponents_pos) == 0:
            return feature

        overall_dist = 1000
        best_opponent = None

        for opponent in opponents_pos:

            x_dist = abs(pos_agent[0] - opponent[0])
            y_dist = abs(pos_agent[1] - opponent[1])
            total_dist = x_dist + y_dist

            # one now needs to account for blocks in the way of the coin
            if x_dist == 0 and (pos_agent[0] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if y_dist == 0 and (pos_agent[1] % 2) == 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if (total_dist < overall_dist):
                overall_dist = total_dist
                best_opponent = opponent

        if pos_agent[0] < best_opponent[0]:
            feature[0] = -1
        elif pos_agent[0] > best_opponent[0]:
            feature[0] = 1

        if pos_agent[1] < best_opponent[1]:
            feature[1] = -1
        elif pos_agent[1] > best_opponent[1]:
            feature[1] = 1

        return feature

    def calc_surrounding_features(self, pos_agent, field, explosion_map, bombs, opponent_position):
        """
        Calculating the surrounding features far a given state
        feature[0] = left
        feature[1] = right
        feature[2] = down
        feature[3] = up
        """
        # self.logger.info("started with the following agent position: "+str(pos_agent))
        feature = np.zeros((8))
        x = pos_agent[0]
        y = pos_agent[1]

        feature[0] = field[x - 1][y]
        feature[1] = field[x + 1][y]
        feature[2] = field[x][y + 1]
        feature[3] = field[x][y - 1]

        if x - 2 < 0 or x - 2 > 16:
            feature[4] = -1
        else:
            feature[4] = field[x - 2][y]

        if x + 2 < 0 or x + 2 > 16:
            feature[5] = -1
        else:
            feature[5] = field[x + 2][y]

        if y - 2 < 0 or y - 2 > 16:
            feature[7] = -1
        else:
            feature[7] = field[x][y - 2]

        if y + 2 < 0 or y + 2 > 16:
            feature[6] = -1
        else:
            feature[6] = field[x][y + 2]

        explosions = np.zeros((8))

        explosions[0] = explosion_map[x - 1][y]
        explosions[1] = explosion_map[x + 1][y]
        explosions[2] = explosion_map[x][y + 1]
        explosions[3] = explosion_map[x][y - 1]
        if x - 2 < 0 or x - 2 > 16:
            explosions[4] = 0
        else:
            explosions[4] = explosion_map[x - 2][y]

        if x + 2 < 0 or x + 2 > 16:
            explosions[5] = 0
        else:
            explosions[5] = explosion_map[x + 2][y]

        if y - 2 < 0 or y - 2 > 16:
            explosions[7] = 0
        else:
            explosions[7] = explosion_map[x][y - 2]

        if y + 2 < 0 or y + 2 > 16:
            explosions[6] = 0
        else:
            explosions[6] = explosion_map[x][y + 2]

        for i in np.arange(len(feature)):
            if int(feature[i]) != -1 and int(explosions[i]) != 0:
                feature[i] = 2
        return feature

    def bombing_effective_feature(self, agent, field, opponent_positions):
        """
        if there is an opponent next to the agent we return 1
        if there is a box next to the agent we return a -1
        else -- or if the agent can currently not place a bomb
        """
        if agent[2] is False:
            return 0

        agent_x, agent_y = agent[3]

        for opponent_position in opponent_positions:
            if abs(agent_x - opponent_position[0]) == 1 and agent_y == opponent_position[1]:
                return 1

            if agent_x == opponent_position[0] and abs(agent_y - opponent_position[1]) == 1:
                return 1

        positions_to_check = [(agent_x + 1, agent_y), (agent_x - 1, agent_y), (agent_x, agent_y - 1),
                              (agent_x, agent_y + 1)]

        for position in positions_to_check:
            if field[position[0]][position[1]] == 1:
                return -1

        return 0

    def bomb_direction_feature(self, pos_agent, bombs):
        '''
        Simple encoding -1, 0, 1 for Left, no_x, Right
        -1, 0, 1 for Up, no_y, Down
        '''
        feature = np.zeros((3))
        if bombs == None:
            feature[2] = -1
            return feature

        overall_dist = 1000
        nearest_bomb = None
        for bomb in bombs:

            x_dist = abs(pos_agent[0] - bomb[0][0])
            y_dist = abs(pos_agent[1] - bomb[0][1])
            total_dist = x_dist + y_dist
            # one now needs to account for blocks in the way of the coin
            if x_dist == 0 and (pos_agent[0] % 2) == 0 and y_dist != 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if y_dist == 0 and (pos_agent[1] % 2) == 0 and x_dist != 0:
                # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                total_dist += 2

            if (total_dist < overall_dist):
                nearest_bomb = bomb[0]
                overall_dist = total_dist
        if nearest_bomb == None:
            return feature
        if overall_dist >= 5:
            return feature
        if pos_agent[0] < nearest_bomb[0]:
            feature[0] = -1
        elif pos_agent[0] > nearest_bomb[0]:
            feature[0] = 1

        if pos_agent[1] < nearest_bomb[1]:
            feature[1] = -1
        elif pos_agent[1] > nearest_bomb[1]:
            feature[1] = 1

        if pos_agent[0] != nearest_bomb[0] and pos_agent[1] != nearest_bomb[1]:
            feature[2] = 0

        # only on a straight number row/column the bomb could be blocked and if they are not on the bomb it is even garantued to
        if pos_agent[0] == nearest_bomb[0] and pos_agent[1] == nearest_bomb[1]:
            feature[2] = 1

        if pos_agent[0] == nearest_bomb[0] and abs(pos_agent[1] - nearest_bomb[1]) < 4:
            if pos_agent[0] % 2 == 0:
                feature[2] = 0
            feature[2] = 1

        if pos_agent[1] == nearest_bomb[1] and abs(pos_agent[0] - nearest_bomb[0]) < 4:
            if pos_agent[1] % 2 == 0:
                feature[2] = 0
            feature[2] = 1

        return feature

    def find_possible_moves(self, position, field):
        x, y = position
        moves = []
        if field[x + 1][y] == 0:
            moves.append("RIGHT")

        if field[x - 1][y] == 0:
            moves.append("LEFT")

        if (field[x][y - 1] == 0):
            moves.appen("UP")

        if (field[x][y + 1] == 0):
            moves.appen("DOWN")

    @staticmethod
    def state_to_index(state):
        """
        Transforming the state variables into indices
        Each state gets a ternary numeral value (Dreierzahlen-system) as index
        """
        temp = np.dot((state + 1), np.array([177147, 59049, 19683, 6561, 2187, 729, 243, 81, 27, 9, 3, 1]))
        return int(np.sum(temp, axis=0))
