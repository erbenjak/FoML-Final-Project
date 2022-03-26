import events as e
import numpy as np
import random
import time

from ..rl_models.BaseQLearningModelExtended import BaseQLearningModel

class ClassicModel(BaseQLearningModel):

    REWARDS = {
        e.INVALID_ACTION : -200,
        ###############
        e.MOVED_UP : -1,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.WAITED: -100,
        ###############
        e.SURVIVED_ROUND: 100,
        e.KILLED_SELF: -1600,
        e.COIN_COLLECTED: 1000,
        e.CRATE_DESTROYED: 200,
        e.KILLED_OPPONENT: 2000,
        e.GOT_KILLED: -700,
        e.COIN_FOUND: 50,
        ################ DISABLED EVENTS ###########################
        e.BOMB_DROPPED : 0,
        e.BOMB_EXPLODED : 0,
        e.OPPONENT_ELIMINATED: 0,
        ############### CUSTOM EVENTS ##############################
        'DEC_COIN_DIST': 100,
        "INC_COIN_DIST": -100,
        'DEC_CRATE_DIST': 90,
        "INC_CRATE_DIST": -90,
        'DEC_BOMB_DIST': -240,
        "INC_BOMB_DIST": 200,
        "KEEP_BOMB_DIST": -200,
        "GOOD_BOMB": 600,
        "VERY_GOOD_BOMB": 800,
        "BAD_BOMB": -800,
        "SURVIVED_BOMB": 100,
        "BOMB_AND_ESCAPE": 1000,
        "BOMB_NO_ESCAPE" : -1400
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
        events = self.addBombDistanceEvents(events,new_state,old_state)
        events = self.addCrateDistanceEvents(events, new_state, old_state)
        events = self.placedBombEvent(events, new_state, old_state)
        events = self.add_escapable_bomb_event(events, new_state, old_state, short_memory)
        return events

    def placedBombEvent(self,events,new_state,old_state):
        agentPos = new_state['self'][3]
        field = new_state['field']

        bomb_positions=[]
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

        if own_bomb==False:
            return events

        # check if a bomb was placed directly besides an opponent
        for opponent in old_state['others']:
            opponent_position = opponent[3]

            if agentPos[0]+1 == opponent_position[0] and agentPos[1] == opponent_position[1]:
                events.append("VERY_GOOD_BOMB")
                return events

            if agentPos[0]-1 == opponent_position[0] and agentPos[1] == opponent_position[1]:
                events.append("VERY_GOOD_BOMB")
                return events

            if agentPos[0] == opponent_position[0] and agentPos[1]+1 == opponent_position[1]:
                events.append("VERY_GOOD_BOMB")
                return events

            if agentPos[0] == opponent_position[0] and agentPos[1]-1 == opponent_position[1]:
                events.append("VERY_GOOD_BOMB")
                return events

        if field[agentPos[0]+1][agentPos[1]] == 1:
            events.append("GOOD_BOMB")
            return events

        if field[agentPos[0]-1][agentPos[1]] == 1:
            events.append("GOOD_BOMB")
            return events

        if field[agentPos[0]][agentPos[1]+1] == 1:
            events.append("GOOD_BOMB")
            return events

        if field[agentPos[0]][agentPos[1]-1] == 1:
            events.append("GOOD_BOMB")
            return events

        events.append("BAD_BOMB")
        return events

    def add_escapable_bomb_event(self,events, new_state, old_state, memory ):
        """
        The idea is to punish any move that would lead to one no loger being able to escape a bomb
        """
        # at least 2 moves need to be played
        if len(memory) < 2:
            return events

        memory_copy = memory.copy()
        self.logger.info("")
        lastMove = memory_copy.popleft()
        secondLastMove = memory_copy.popleft()

        if secondLastMove[1] != "BOMB":
            return events

        if new_state["self"][3] == old_state["self"][3]:
            events.append("BOMB_NO_ESCAPE")
            return events

        feature_escape_bomb = self.calculateFeaturesFromState(new_state)[6:10]
        not_escaped = np.all(feature_escape_bomb == -1)

        if not_escaped:
            events.append("BOMB_NO_ESCAPE")
        else:
            events.append("BOMB_AND_ESCAPE")

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

        bombPosOld=[]
        for bomb in old_state['bombs']:
            bombPosOld.append(bomb[0])

        bombPosNew=[]
        for bomb in new_state['bombs']:
            bombPosNew.append(bomb[0])

        old_distance, closestBombPosOld = self.calc_clossest_bomb_dist(old_state['self'][3], bombPosOld)
        new_distance, closestBombPosNew = self.calc_clossest_bomb_dist(new_state['self'][3], bombPosNew)

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
        if(e.COIN_COLLECTED in events):
            return events

        #since no coin was present the agent could not possibly moce towards it
        if len(old_state['coins']) == 0:
            return events

        old_distance =  self.calc_coin_dist(old_state['self'][3], old_state['coins'])
        new_distance =  self.calc_coin_dist(new_state['self'][3], new_state['coins'])
        if old_distance < new_distance :
            events.append("INC_COIN_DIST")
            return events

        if new_distance < old_distance :
            events.append("DEC_COIN_DIST")
            return events

        return events

    def addCrateDistanceEvents(self, events, new_state, old_state):
        crates_old = np.argwhere(old_state['field'] == 1)
        crates_new = np.argwhere(new_state['field'] == 1)
        #print("crates old position:" + str(crates_old))
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
        #print("crate dist:" + str(overall_dist))
        return overall_dist

    @staticmethod
    def calc_clossest_bomb_dist(agent, bombpositions):
        if (len(bombpositions) == 0):
            return 0, None
        overall_dist = 1000
        bestBombPos = None

        for bombpos in bombpositions:

            x_dist = abs(agent[0] - bombpos[0])
            y_dist = abs(agent[1] - bombpos[0])
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
                bestBombPos = bombpos

        return overall_dist, bestBombPos

    @staticmethod
    def calc_bomb_distance(x, y, single_bomb_pos):
        x_dist = abs(x - single_bomb_pos[0])
        y_dist = abs(y - single_bomb_pos[0])
        total_dist = x_dist + y_dist
        # one now needs to account for blocks in the way of the coin

        if x_dist == 0 and (x % 2) == 0 and y_dist != 0:
            # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
            total_dist += 2

        if y_dist == 0 and (y % 2) == 0 and x_dist != 0:
            # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
            total_dist += 2

        return total_dist


    @staticmethod
    def calc_coin_dist(agent,coins):
        if(len(coins)==0):
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
        self.logger.info(str(events))
        for event in events:
            reward += self.REWARDS[event]
        return reward

    def calculateFeaturesFromState(self, state):
        """
        Takes a gamestate and extracts the features which are required to take good decisions.
        """
        encoded_state = np.array([])

        #some minor data prep for the calculating methods
        pos_agent = (state['self'])[3]
        coins = state['coins']
        field = state['field']
        explosionMap = state['explosion_map']

        bombPositions = []
        for bomb in state ["bombs"]:
            bombPositions.append(bomb[0])

        distance, closestBombPos = self.calc_clossest_bomb_dist(pos_agent, bombPositions)

        opponent_positions=[]
        for opponent in state['others']:
            opponent_positions.append(opponent[3])

        # adds two features
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_coin(pos_agent, coins, field))
        # adds four feature
        encoded_state = np.append(encoded_state, self.calc_surrounding_features(pos_agent, field, explosionMap))
        # adds four feature
        encoded_state = np.append(encoded_state, self.calc_surrounding_escapability(pos_agent, field, closestBombPos,bombPositions, opponent_positions))
        #adds one feature
        encoded_state = np.append(encoded_state, self.bombing_effective_feature(state['self'],field,opponent_positions))
        # adds one feature
        encoded_state = np.append(encoded_state, self.bomb_explodable_feature(pos_agent, closestBombPos))
        # adds two features
        encoded_state = np.append(encoded_state, self.bomb_direction_feature(pos_agent, closestBombPos))
        # adds two features
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_crate(pos_agent, field))
        # adds two features
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_opponent(pos_agent, field, opponent_positions))
        return encoded_state

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
                total_dist = x_dist+y_dist

                # one now needs to account for blocks in the way of the coin
                if x_dist==0 and (pos_agent[0] % 2) == 0:
                    #then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                    total_dist += 2

                if y_dist == 0 and (pos_agent[1] % 2) == 0:
                    # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                    total_dist += 2

                if(total_dist < overall_dist):
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
                self.logger.info("No more crates left to destroy ")
                return feature

            for crate in crates:
                x_dist = abs(pos_agent[0] - crate[0])
                y_dist = abs(pos_agent[1] - crate[1])
                total_dist = x_dist+y_dist

                # one now needs to account for blocks in the way of the coin
                if x_dist==0 and (pos_agent[0] % 2) == 0:
                    #then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                    total_dist += 2

                if y_dist == 0 and (pos_agent[1] % 2) == 0:
                    # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                    total_dist += 2

                if(total_dist < overall_dist):
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
                            -1, 0, 1 for Down, no_y, Up
            '''
            feature = np.zeros((2))
            overall_dist = 1000
            best_opponent = None

            if len(opponents_pos) == 0:
                return feature

            for opponent in opponents_pos:

                x_dist = abs(pos_agent[0] - opponent[0])
                y_dist = abs(pos_agent[1] - opponent[1])
                total_dist = x_dist+y_dist

                # one now needs to account for blocks in the way of the coin
                if x_dist==0 and (pos_agent[0] % 2) == 0:
                    #then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                    total_dist += 2

                if y_dist == 0 and (pos_agent[1] % 2) == 0:
                    # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
                    total_dist += 2

                if(total_dist < overall_dist):
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

    def calc_surrounding_features(self,pos_agent, field, explosion_map):
        """
        Calculating the surrounding features far a given state

        feature[0] = left
        feature[1] = right
        feature[2] = down
        feature[3] = up
        """
        #self.logger.info("started with the following agent position: "+str(pos_agent))
        feature = np.zeros((4))
        x = pos_agent[0]
        y = pos_agent[1]

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

        return feature

    def calc_surrounding_escapability(self,pos_agent, field, closestBombPos, bombPositions, opponentPositions):
        """
        Calculating the bomb escape-capability for surrounding fields
        --> Meaning is it possible to escape the closest bomb from there
        0   - no bomb or the agent is already escaped
        1   - bomb and helps escape the agent
        -1  - bomb and does not allow for escaping
        feature[0] = left
        feature[1] = right
        feature[2] = down
        feature[3] = up
        """
        #self.logger.info("started with the following agent position: "+str(pos_agent))
        feature = np.zeros((4))

        if closestBombPos is None:
            return feature

        x = pos_agent[0]
        y = pos_agent[1]

        x_bomb = closestBombPos[0]
        y_bomb = closestBombPos[1]

        feature[0] = -abs(field[x - 1][y])
        feature[1] = -abs(field[x + 1][y])
        feature[2] = -abs(field[x][y - 1])
        feature[3] = -abs(field[x][y + 1])

        currently_in_danger = self.inExplosionZone(x_bomb,y_bomb,x,y)

        # if the agent is currently not in danger it is checked if one of his steps leads him into danger
        if currently_in_danger is False:
            if self.inExplosionZone(x_bomb, y_bomb, x-1, y):
                feature[0] = -1

            if self.inExplosionZone(x_bomb, y_bomb, x+1, y):
                feature[1] = -1

            if self.inExplosionZone(x_bomb, y_bomb, x, y-1):
                feature[2] = -1

            if self.inExplosionZone(x_bomb, y_bomb, x, y+1):
                feature[3] = -1

            return feature

        #if a feature can be ruled out it is removed from the list and it's value is saved
        feature_to_consider = [0, 1, 2, 3]

        # check is any fields are not free
        if not self.coordinates_are_free(x-1,y,bombPositions,field, opponentPositions):
            feature[0] = -1
            feature_to_consider.remove(0)

        if not self.coordinates_are_free(x+1,y,bombPositions,field, opponentPositions):
            feature[1] = -1
            feature_to_consider.remove(1)

        if not self.coordinates_are_free(x,y-1,bombPositions,field, opponentPositions):
            feature[2] = -1
            feature_to_consider.remove(2)

        if not self.coordinates_are_free(x,y+1,bombPositions,field, opponentPositions):
            feature[3] = -1
            feature_to_consider.remove(3)

        #return if all directions are blocked
        if len(feature_to_consider) == 0:
            return feature

        # check if any of the surrounding fields allows for immediate escaping
        if 0 in feature_to_consider:
            if not self.inExplosionZone(x_bomb, y_bomb, x - 1, y):
                feature[0] = 1
                feature_to_consider.remove(0)

        if 1 in feature_to_consider:
            if not self.inExplosionZone(x_bomb, y_bomb, x + 1, y):
                feature[1] = 1
                feature_to_consider.remove(1)

        if 2 in feature_to_consider:
            if not self.inExplosionZone(x_bomb, y_bomb, x, y - 1):
                feature[2] = 1
                feature_to_consider.remove(2)

        if 3 in feature_to_consider:
            if not self.inExplosionZone(x_bomb, y_bomb, x, y + 1):
                feature[3] = 1
                feature_to_consider.remove(3)

        # return if all directions are considered
        if len(feature_to_consider) == 0:
            return feature

        # check running away in a straight line
        for i in range(1,5):
            if len(feature_to_consider) == 0:
                break
            # running left
            if 0 in feature_to_consider:
                # hitting a stop
                if not self.coordinates_are_free(x-i, y, bombPositions, field, opponentPositions):
                    feature[0] = -1
                    feature_to_consider.remove(0)
                else:
                    if not self.inExplosionZone(x_bomb, y_bomb, x - i, y):
                        feature[0] = 1
                        feature_to_consider.remove(0)
                    else:
                        step_up_possible = self.coordinates_are_free(x-i, y+1, bombPositions, field, opponentPositions)
                        step_down_possible = self.coordinates_are_free(x-i, y-1, bombPositions, field, opponentPositions)

                        if step_up_possible or step_down_possible:
                            feature[0] = 1
                            feature_to_consider.remove(0)

            # running right
            if 1 in feature_to_consider:
                # hitting a stop
                if not self.coordinates_are_free(x + i, y, bombPositions, field, opponentPositions):
                    feature[1] = -1
                    feature_to_consider.remove(1)
                else:
                    if not self.inExplosionZone(x_bomb, y_bomb, x + i, y):
                        feature[1] = 1
                        feature_to_consider.remove(1)
                    else:
                        step_up_possible = self.coordinates_are_free(x + i, y + 1, bombPositions, field,
                                                                     opponentPositions)
                        step_down_possible = self.coordinates_are_free(x + i, y - 1, bombPositions, field,
                                                                       opponentPositions)

                        if step_up_possible or step_down_possible:
                            feature[1] = 1
                            feature_to_consider.remove(1)

            # running up
            if 2 in feature_to_consider:
                # hitting a stop
                if not self.coordinates_are_free(x, y - i, bombPositions, field, opponentPositions):
                    feature[2] = -1
                    feature_to_consider.remove(2)
                else:
                    if not self.inExplosionZone(x_bomb, y_bomb, x, y - i):
                        feature[2] = 1
                        feature_to_consider.remove(2)
                    else:
                        step_left_possible = self.coordinates_are_free(x - 1, y - i, bombPositions, field,
                                                                     opponentPositions)
                        step_right_possible = self.coordinates_are_free(x + 1, y - i, bombPositions, field,
                                                                       opponentPositions)
                        if step_left_possible or step_right_possible:
                            feature[2] = 1
                            feature_to_consider.remove(2)

            # running down
            if 3 in feature_to_consider:
                # hitting a stop
                if not self.coordinates_are_free(x, y + i, bombPositions, field, opponentPositions):
                    feature[3] = -1
                    feature_to_consider.remove(3)
                else:
                    if not self.inExplosionZone(x_bomb, y_bomb, x, y + i):
                        feature[3] = 1
                        feature_to_consider.remove(3)
                    else:
                        step_left_possible = self.coordinates_are_free(x - 1, y + i, bombPositions, field,
                                                                     opponentPositions)
                        step_right_possible = self.coordinates_are_free(x + 1, y + i, bombPositions, field,
                                                                       opponentPositions)
                        if step_left_possible or step_right_possible:
                            feature[3] = 1
                            feature_to_consider.remove(3)

        self.logger.info("Calculated the following Escape-feature: " + str(feature))

        if len(feature_to_consider) != 0:
            raise ValueError("List of features to consider should be empty but is not: "+ str(len(feature_to_consider)))

        return feature

    @staticmethod
    def inExplosionZone(x_bomb, y_bomb, pos_x, pos_y):
        #on two different axes
        if x_bomb != pos_x and y_bomb != pos_y:
            return False

        #identical
        if x_bomb == pos_x and y_bomb == pos_y:
            return True

        if y_bomb == pos_y:
            if y_bomb % 2 == 0:
                return False

            if abs(x_bomb - pos_x) > 3:
                return False

            return True

        if x_bomb == pos_x:
            if x_bomb % 2 == 0:
                return False

            if abs(y_bomb - pos_y) > 3:
                return False

            return True

    @staticmethod
    def coordinates_are_free(x,y,bomb_positions ,field, opponent_positions):

        if(field[x][y]!=0):
            return False

        for single_bomb_pos in bomb_positions:
            if single_bomb_pos[0]==x and single_bomb_pos[1]==y:
                return False

        for single_opponent_pos in opponent_positions:
            if single_opponent_pos[0]==x and single_opponent_pos[1]==y:
                return False

        # field is free
        return True

    def bombing_effective_feature(self,agent,field,opponent_positions):
        """
        if there is an opponent next to the agent we return 1
        if there is a box next to the agent we return a -1
        else -- or if the agent can currently not place a bomb
        """
        if agent[2] is False:
            return 0

        agent_x, agent_y = agent[3]

        for opponent_position in opponent_positions:
            if abs(agent_x-opponent_position[0]) == 1 and agent_y==opponent_position[1]:
                return 1

            if agent_x==opponent_position[0] and abs(agent_y-opponent_position[1]) == 1:
                return 1

        positions_to_check=[(agent_x+1,agent_y),(agent_x-1,agent_y),(agent_x,agent_y-1),(agent_x,agent_y+1)]

        for position in positions_to_check:
            if  field[position[0]][position[1]] == 1:
                return -1

        return 0


    def bomb_direction_feature(self, pos_agent, bomb):
        '''
        Simple encoding -1, 0, 1 for Left, no_x, Right
        -1, 0, 1 for Up, no_y, Down
        '''
        feature = np.zeros((2))

        if bomb is None:
            return feature

        if pos_agent[0] < bomb[0]:
            feature[0] = -1
        elif pos_agent[0] > bomb[0]:
            feature[0] = 1

        if pos_agent[1] < bomb[1]:
            feature[1] = -1
        elif pos_agent[1] > bomb[1]:
            feature[1] = 1
        return feature

    def bomb_explodable_feature(self, pos_agent, bomb):
        """
        -1 no bomb
        0 not in bomb radius
        1 inside bomb radius
        """
        if bomb is None:
            return -1

        if pos_agent[0] != bomb[0] and pos_agent[1] != bomb[1]:
            return 0

        # only on a straight number row/column the bomb could be blocked and if they are not on the bomb it is even garantued to
        if pos_agent[0] == bomb[0] and pos_agent[1] == bomb[1]:
            return 1

        if pos_agent[0] == bomb[0] and abs(pos_agent[1]-bomb[1])<4:
            if pos_agent[0] % 2 == 0:
                return 0
            return 1

        if pos_agent[1] == bomb[1] and abs(pos_agent[0]-bomb[0])<4:
            if pos_agent[1] % 2 == 0:
                return 0
            return 1

        return 0

    def find_possible_moves(self, position, field):
        x,y = position
        moves = []
        if field [x+1][y] == 0:
            moves.append("RIGHT")

        if field [x-1][y] == 0:
            moves.append("LEFT")

        if(field [x][y-1] == 0):
            moves.appen("UP")

        if (field[x][y+1] == 0):
            moves.appen("DOWN")


    @staticmethod
    def state_to_index(state):
        """
        Transforming the state variables into indices

        Each state gets a ternary numeral value (Dreierzahlen-system) as index
        """
        temp = np.dot((state + 1), np.array([177147, 59049, 19683, 6561, 2187, 729, 243, 81, 27, 9, 3, 1]))
        return int(np.sum(temp, axis=0))