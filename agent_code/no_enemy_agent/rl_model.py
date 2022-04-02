import events as e
import numpy as np
import random
from ..rl_models.BaseQLearningModel import BaseQLearningModel

class RLModel(BaseQLearningModel):
    """
    This class represents the regression model and will therefor keep track of its current features and
    will be calculating the rewards for when an event occurs

    REMINDER: x- and y-axis of the board point right and down (matrix-view)
                they do NOT point right and up (typical physical view)
    """

    def __init__(self, logger, max_feature_size, path):
        BaseQLearningModel.__init__(self, logger, max_feature_size, path)

    def compute_additional_rewards(self, events, new_state, old_state):
        """
        Takes a set of events produced by the game engine and adds some custom events to be able to
        add some additional self-defined 'custom' events
        """
        events = self.addCoinDistanceEvents(events,new_state,old_state)
        events = self.addBombDistanceEvents(events, new_state, old_state)
        return events

    def addCoinDistanceEvents(self, events, new_state, old_state):
        # if a coin was collected the distance was shortened:
        if(e.COIN_COLLECTED in events):
            return events

        old_distance =  self.calc_coin_dist(old_state['self'][3], old_state['coins'])
        new_distance =  self.calc_coin_dist(new_state['self'][3], new_state['coins'])
        if old_distance < new_distance :
            events.append("INC_COIN_DIST")
            return events

        if new_distance < old_distance :
            events.append("DEC_COIN_DIST")
            return events

        events.append("KEEP_COIN_DIST")
        return events

    def addBombDistanceEvents(self, events, new_state, old_state):
        # if a coin was collected the distance was shortened:
        old_distance =  self.calc_bomb_dist(old_state['self'][3], old_state['bombs'])
        new_distance =  self.calc_bomb_dist(new_state['self'][3], new_state['bombs'])
        if old_distance < new_distance :
            events.append("INC_BOMB_DIST")
            return events

        if new_distance < old_distance :
            events.append("DEC_BOMB_DIST")
            return events

        if new_distance == old_distance:
            events.append("KEEP_BOMB_DIST")
            return events
        return events

    def calculateFeaturesFromState(self, state):
        # just want the direction to the nearest coin and the surrounding moving options as features
        # [CoinUpDown][CoinLeftRight][LeftWallNoWall][RightWallNoWall][UpWallNoWall][DownWallNoWall]
        encoded_state = np.array([])
        pos_agent = (state['self'])[3]
        bomb_availibility = int(state['self'][2] == True)-1
        coins = state['coins']
        field = state['field']
        bombs = state['bombs']
        explosion_map = state['explosion_map']

        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_coin(pos_agent, coins, field, bombs, explosion_map))
        encoded_state = np.append(encoded_state, self.calc_surrounding_features(pos_agent, field, bombs, explosion_map))
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_bomb(pos_agent, field, bombs, explosion_map))
        encoded_state = np.append(encoded_state, bomb_availibility)

        return encoded_state

    @staticmethod
    def calc_coin_dist(agent, coins):
        if (len(coins) == 0):
            return 0

        for coin in coins:
            overall_dist = 1000

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

    @staticmethod
    def calc_bomb_dist(agent, bombs):
        if (len(bombs) == 0):
            return 0

        for bomb in bombs:
            overall_dist = 1000

            x_dist = abs(agent[0] - bomb[0][0])
            y_dist = abs(agent[1] - bomb[0][1])
            total_dist = x_dist + y_dist

            if (total_dist < overall_dist):
                overall_dist = total_dist
        return overall_dist

    def calc_distance_and_direction_coin(self, pos_agent, coins, field, bombs, explosion_map):
        # picking some very long distance
        best_distance = 1000.0
        hor_distance = 1000.0
        ver_distance = 1000.0
        closest_coin = None
        # if there is no more coin on the board, return zeros
        if len(coins) == 0:
            return np.zeros((3))
        else:
            for coin in coins:
                '''
                calculating the horizontal and vertical distance between the coin and the agent
                the distance is the sum of horizontal and vertical distance, since the agent cant move diagonal
                '''
                horizontal_distance = np.abs(coin[0] - pos_agent[0])
                vertical_distance = np.abs(coin[1] - pos_agent[1])
                distance_coin = horizontal_distance + vertical_distance
                surroundings = self.calc_surrounding_features(pos_agent, field, bombs, explosion_map)
                '''
                If the direct path is blocked by a boulder, the efficient distance
                gets larger by the value=2.
                In the case where one distance-parameter shows the value 0 and the surroundings in the other dimension
                are boulders we have to add 2 to the actual distance value
                '''
                if (int(horizontal_distance) == 0) and (int(surroundings[2]) == -1) and (int(surroundings[3]) == -1):
                    distance_coin += 2
                if (int(vertical_distance) == 0) and (int(surroundings[0]) == -1) and (int(surroundings[1]) == -1):
                    distance_coin += 2
                '''
                Updating the best coin and the relative parameters
                '''
                if distance_coin < best_distance:
                    best_distance = distance_coin
                    closest_coin = coin
                    hor_distance = horizontal_distance
                    ver_distance = vertical_distance
            '''
            Setting the features

            first feature dimension describes the relative distance along the x-axis:
                1.0     = nearest coin lies left of agent
                0.0     = nearest coin lies in the same column
                -1.0    = nearest coin lies right of agent

            second feature dimension describes the relative distance along the y-axis:
                1.0     = nearest coin lies above the agent
                0.0     = nearest coin lies in the same row
                -1.0    = nearest coin lies under the agent

            third feature dimension describes the relative magnitude between the horizontal and vertical distance to the nearest coin
                1.0     = vertical distance > horizontal distance
                0.0     = vertical distance = horizontal distance
                -1.0    = vertical distance < horizontal distance
            '''
            feature = np.zeros((3))

            if pos_agent[0] < closest_coin[0]:
                feature[0] = -1
            elif pos_agent[0] > closest_coin[0]:
                feature[0] = 1

            if pos_agent[1] < closest_coin[1]:
                feature[1] = -1
            elif pos_agent[1] > closest_coin[1]:
                feature[1] = 1

            if hor_distance < ver_distance:
                feature[2] = -1
            elif hor_distance > ver_distance:
                feature[2] = 1

            '''
            some debug stuff
            '''
            # print("closest coin:" +str(closest_coin)+"distance:"+str(best_distance))
            # print("horizontal_distance:"+str(hor_distance),"vertical_distance:"+str(ver_distance))
            return feature

    def calc_distance_and_direction_bomb(self, pos_agent, field, bombs, explosion_map):
        # picking some very long distance
        best_distance = 1000.0
        hor_distance = 1000.0
        ver_distance = 1000.0
        closest_bomb = None
        # if there is no more coin on the board, return zeros
        if len(bombs) == 0:
            return np.zeros((3))
        else:
            for bomb in bombs:
                '''
                calculating the horizontal and vertical distance between the bomb and the agent
                the distance is the sum of horizontal and vertical distance, since the agent cant move diagonal
                '''
                horizontal_distance = np.abs(bomb[0][0] - pos_agent[0])
                vertical_distance = np.abs(bomb[0][1] - pos_agent[1])
                distance_bomb = horizontal_distance + vertical_distance
                surroundings = self.calc_surrounding_features(pos_agent, field, bombs, explosion_map)
                '''
                If the direct path is blocked by a boulder, the efficient distance
                gets larger by the value=2.
                In the case where one distance-parameter shows the value 0 and the surroundings in the other dimension
                are boulders we have to add 2 to the actual distance value
                '''
                if (int(horizontal_distance) == 0) and (int(surroundings[2]) == -1) and (int(surroundings[3]) == -1):
                    distance_bomb += 2
                if (int(vertical_distance) == 0) and (int(surroundings[0]) == -1) and (int(surroundings[1]) == -1):
                    distance_bomb += 2
                '''
                Updating the best bomb and the relative parameters
                '''
                if distance_bomb < best_distance:
                    best_distance = distance_bomb
                    closest_bomb = bomb
                    hor_distance = horizontal_distance
                    ver_distance = vertical_distance
            '''
            Setting the features

            first feature dimension describes the relative distance along the x-axis:
                1.0     = nearest bomb lies left of agent
                0.0     = nearest bomb lies in the same column
                -1.0    = nearest bomb lies right of agent

            second feature dimension describes the relative distance along the y-axis:
                1.0     = nearest bomb lies above the agent
                0.0     = nearest bomb lies in the same row
                -1.0    = nearest bomb lies under the agent

            third feature dimension describes the relative magnitude between the horizontal and vertical distance to the
            nearest bomb
                1.0     = vertical distance > horizontal distance
                0.0     = vertical distance = horizontal distance
                -1.0    = vertical distance < horizontal distance
            '''
            feature = np.zeros((3))

            if pos_agent[0] < closest_bomb[0][0]:
                feature[0] = -1
            elif pos_agent[0] > closest_bomb[0][0]:
                feature[0] = 1

            if pos_agent[1] < closest_bomb[0][1]:
                feature[1] = -1
            elif pos_agent[1] > closest_bomb[0][1]:
                feature[1] = 1

            if hor_distance < ver_distance:
                feature[2] = -1
            elif hor_distance > ver_distance:
                feature[2] = 1

            '''
            some debug stuff
            '''
            # print("closest coin:" +str(closest_coin)+"distance:"+str(best_distance))
            # print("horizontal_distance:"+str(hor_distance),"vertical_distance:"+str(ver_distance))
            return feature

    def calc_surrounding_features(self, pos_agent, field, bombs, explosion_map):
        """
        Calculating the surrounding features far a given state

        feature[0] = left
        feature[1] = right
        feature[2] = down
        feature[3] = up
        """
        feature = np.zeros((4))
        x = pos_agent[0]
        y = pos_agent[1]
        explosions = np.zeros((4))
        for bomb in bombs:

            if int(bomb[0][1]) == int(y):
                if np.abs(x - 1 - bomb[0][0]) == 0:
                    explosions[0] = 2
                if np.abs(x + 1 - bomb[0][0]) == 0:
                    explosions[1] = 2
            if int(bomb[0][0]) == int(x):
                if np.abs(y + 1 - bomb[0][1]) == 0:
                    explosions[2] = 2
                if np.abs(y - 1 - bomb[0][1]) == 0:
                    explosions[3] = 2




        feature[0] = field[x - 1][y]
        feature[1] = field[x + 1][y]
        feature[2] = field[x][y + 1]
        feature[3] = field[x][y - 1]

        for i,feat in enumerate(feature):
            if int(feat) == 0:
                feature[i] = explosions[i]
        return feature

    def calculateReward(self, events):
        """
        Calculating rewards:
        """

        """
        Negative reward for each move to minimize the game-time needed
        """
        rewards = -50
        for event in events:
            if event == e.COIN_COLLECTED:
                rewards += 50
            """
            Rewarding the move commands:

            reducing the distance to the nearest coin is rewarded positively
            while increasing the distance is rewarded negatively

            reducing the distance in the direction of biggest gradient is rewarded
            more positively since it decreases the chance of getting blocked by a boulder
            """
            if event in [e.MOVED_UP, e.MOVED_DOWN, e.MOVED_LEFT, e.MOVED_RIGHT]:
                rewards -= 50
            if event in [e.WAITED]:
                rewards -= 50
            if event == e.INVALID_ACTION:
                rewards -= 500
            if event == 'DEC_COIN_DIST':
                self.logger.info("decreased coin distance event")
                rewards += 50
            if event == "INC_COIN_DIST":
                self.logger.info("increased coin distance event")
                rewards -= 50
            if event == e.KILLED_SELF:
                rewards -= 500
            if event == e.KILLED_OPPONENT:
                rewards += 50
            if event == e.GOT_KILLED:
                rewards -= 500
            if event == e.OPPONENT_ELIMINATED:
                rewards += 50
            if event == e.SURVIVED_ROUND:
                rewards += 50
            if event == e.BOMB_DROPPED:
                rewards += 50
            if event == e.BOMB_EXPLODED:
                rewards += 50
            if event == e.COIN_FOUND:
                rewards += 50
            if event == e.CRATE_DESTROYED:
                rewards += 50

            if event == 'DEC_BOMB_DIST':
                rewards -= 100

            if event == 'INC_BOMB_DIST':
                rewards += 100

            if event == 'KEEP_BOMB_DIST':
                rewards -= 50
        """
        Some debug stuff
        """
        # print("HOR_DISTANCE" +str(self.HOR_DISTANCE),"VER_DISTANCE" +str(self.VER_DISTANCE))
        # print("Route:" + str(self.ROUTE_PLANER))
        # print("Rewards" +str(rewards))
        return rewards

    def create_qtable(self):
        """
        creating a q-table of need size

        here we need 2187 rows, since there are 7 features which all can hold 3 different values
            -> 3^7 = 2187

        and we need 6 columns, since there are 6 possible actions in this scenario
        """
        self.Q_TABLE = np.zeros((373248, 6))
        return

    def get_qtable(self):
        return self.Q_TABLE

    def set_qtable(self, qtable):
        self.Q_TABLE = qtable
        return

    def get_epsilon(self):
        return self.EPSILON

    @staticmethod
    def state_to_index(state):
        """
        Transforming the state variables into indices
        """
        temp = np.dot((state + 1), np.array([1, 3, 9, 27, 108, 432, 1728, 6912, 20736, 61208, 186624]))
        return int(np.sum(temp, axis=0))