import events as e
import numpy as np
import random
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
        e.KILLED_SELF: -1000,
        e.COIN_COLLECTED: 300,
        e.CRATE_DESTROYED: 200,
        e.KILLED_OPPONENT: 1000,
        e.GOT_KILLED: -700,
        e.COIN_FOUND: 50,
        ################ DISABLED EVENTS ###########################
        e.BOMB_DROPPED : 0,
        e.BOMB_EXPLODED : 0,
        ############### CUSTOM EVENTS ##############################
        'DEC_COIN_DIST': 80,
        "INC_COIN_DIST": -80,
        'DEC_CRATE_DIST': 60,
        "INC_CRATE_DIST": -60,
        'DEC_BOMB_DIST': -100,
        "INC_BOMB_DIST": 100,
        "KEEP_BOMB_DIST": -20,
        "GOOD_BOMB": 600,
        "BAD_BOMB": -800,
        "SURVIVED_BOMB": 100,
    }



    """
    This is an implementation of the BaseQLearningModel.

    REMINDER: x- and y-axis of the board point right and down (matrix-view)
                they do NOT point right and up (typical physical view)
    """

    def __init__(self, logger, max_feature_size, path_q_table, path_seen_representations):
        BaseQLearningModel.__init__(self, logger, max_feature_size, path_q_table, path_seen_representations)

    ############### Necessary Overwrites ##############################################
    def compute_additional_rewards(self, events, new_state, old_state):
        """
        Takes a set of events produced by the game engine and adds some custom events to be able to
        add some additional self-defined 'custom' events
        """
        events = self.addSurrivedABombEvent(events,old_state)
        events = self.addCoinDistanceEvents(events, new_state, old_state)
        events = self.addBombDistanceEvents(events,new_state,old_state)
        events = self.placedBombEvent(events,new_state)
        return events

    def placedBombEvent(self,events,new_state):
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

        if len(new_state["bombs"]) > 1:
            raise NotImplementedError("add bomb distance events can't handle multiple bombs so far")

        clossestBombOld = old_state["bombs"][0]
        clossestBombCoordinatesOld = clossestBombOld[0]

        clossestBombNew = new_state["bombs"][0]
        clossestBombCoordinatesNew = clossestBombNew[0]

        old_distance = self.calc_bomb_dist(old_state['self'][3], clossestBombCoordinatesOld)
        new_distance = self.calc_bomb_dist(new_state['self'][3], clossestBombCoordinatesNew)

        self.logger.info("old_distance: "+  str(old_distance))
        self.logger.info("new_distance: "+  str(new_distance))

        if old_distance < new_distance:
            events.append("INC_BOMB_DIST")
            return events

        if new_distance < old_distance:
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
    def calc_bomb_dist(agent, crates):
        if (len(crates) == 0):
            return 0

        for crate in crates:
            overall_dist = 1000

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
        return overall_dist

    @staticmethod
    def calc_bomb_dist(agent,bomb):

        x_dist = abs(agent[0] - bomb[0])
        y_dist = abs(agent[1] - bomb[1])
        total_dist = x_dist + y_dist
        # one now needs to account for blocks in the way of the coin

        if x_dist == 0 and (agent[0] % 2) == 0:
            # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
            total_dist += 2

        if y_dist == 0 and (agent[1] % 2) == 0:
            # then coin and agent are on the same column, which also has walls --> therefore increase total by 2
            total_dist += 2

        return total_dist


    @staticmethod
    def calc_coin_dist(agent,coins):
        if(len(coins)==0):
            return 0

        for coin in coins:
            overall_dist=1000

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
        for event in events:
            reward += self.REWARDS[event]
            if event == e.SURVIVED_ROUND:
                self.logger.info("STAYING ALIVEEEEEEEEEEEEEEEEE")
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
        bombPos = None

        if len(state["bombs"]) > 0:
            bombPos = state["bombs"][0][0]

        opponent_positions=[]
        for opponent in state['others']:
            opponent_positions.append(opponent[3])

        # adds two features
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_coin(pos_agent, coins, field))
        # adds four feature
        encoded_state = np.append(encoded_state, self.calc_surrounding_features(pos_agent, field))
        #adds one feature
        encoded_state = np.append(encoded_state, self.bombing_effective_feature(state['self'],field,opponent_positions))
        # adds one feature
        encoded_state = np.append(encoded_state, self.bomb_explodable_feature(pos_agent, bombPos))
        # adds two features
        encoded_state = np.append(encoded_state, self.bomb_direction_feature(pos_agent, bombPos))
        # adds two features
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_crate(pos_agent, field))
        return encoded_state

    def calc_distance_and_direction_coin(self, pos_agent, coins, field):
            '''
            Simple encoding -1, 0, 1 for Left, no_x, Right
                            -1, 0, 1 for Down, no_y, Up
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

            '''
            some debug stuff
            '''
            # print("closest coin:" +str(closest_coin)+"distance:"+str(best_distance))
            # print("horizontal_distance:"+str(hor_distance),"vertical_distance:"+str(ver_distance))
            return feature

    def calc_distance_and_direction_crate(self, pos_agent, field):
            '''
            Simple encoding -1, 0, 1 for Left, no_x, Right
                            -1, 0, 1 for Down, no_y, Up
            '''
            feature = np.zeros((2))
            overall_dist = 1000
            best_crate = None

            crates = np.argwhere(field == 1)

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

            '''
            some debug stuff
            '''
            # print("closest coin:" +str(closest_coin)+"distance:"+str(best_distance))
            # print("horizontal_distance:"+str(hor_distance),"vertical_distance:"+str(ver_distance))
            return feature

    def calc_surrounding_features(self,pos_agent, field):
        """
        Calculatiing the surrounding features far a given state

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
        return feature

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

            if agent_x==opponent_position[0] and abs(agent_x-opponent_position[0]) == 1:
                return 1

        positions_to_check=[(agent_x+1,agent_y),(agent_x-1,agent_y),(agent_x,agent_y+1),(agent_x,agent_y+1)]

        for position in positions_to_check:
            if  field[position[0]][position[1]] == 1:
                return -1

        return 0


    def bomb_direction_feature(self, pos_agent, bomb):
        '''
        Simple encoding -1, 0, 1 for Left, no_x, Right
        -1, 0, 1 for Down, no_y, Up
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
        1 inside bomb raius
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
        moves=[]
        if(field [x+1][y] == 0):
            moves.append("RIGHT")

        if(field [x-1][y] == 0):
            moves.append("LEFT")

        if(field [x][y+1] == 0):
            moves.appen("UP")


    @staticmethod
    def state_to_index(state):
        """
        Transforming the state variables into indices

        Each state gets a ternary numeral value (Dreierzahlen-system) as index
        """
        temp = np.dot((state + 1), np.array([177147, 59049, 19683, 6561, 2187, 729, 243, 81, 27, 9, 3, 1]))
        return int(np.sum(temp, axis=0))