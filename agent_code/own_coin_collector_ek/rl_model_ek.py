import events as e
import numpy as np
import random


class CoinRlModel:
    """
    This class represents the regression model and will therefor keep track of its current features and
    will be calculating the rewards for when an event occurs

    REMINDER: x- and y-axis of the board point right and down (matrix-view)
                they do NOT poitn right and up (typical physical view)
    """
    ACTIONS = []
    Q_TABLE = np.array([])
    state_size = -1
    """defining the hyper-parameters"""
    #Learning rate
    ALPHA = 0.15
    #Discount factor
    GAMMA = 0.10
    #Probability to random move -> e-greedy policy
    EPSILON = 0.05

    """initializing some training parameters"""
    """
    HOR_DISTANCE describes the relative distance along the x-axis:
        1.0     = nearest coin lies left of agent
        0.0     = nearest coin lies in the same column
        -1.0    = nearest coin lies right of agent
    """
    HOR_DISTANCE = 1000.0
    """
    VER_DISTANCE describes the relative distance along the y-axis:
        1.0     = nearest coin lies above the agent
        0.0     = nearest coin lies in the same row
        -1.0    = nearest coin lies under the agent
    """
    VER_DISTANCE = 1000.0
    """
    ROUTE_PLANER describes the relative magnitude between the horizontal and vertical distance to the nearest coin
        1.0     = vertical distance > horizontal distance
        0.0     = vertical distance = horizontal distance
        -1.0    = vertical distance < horizontal distance
    """
    ROUTE_PLANER = 1000.0

    def __init__(self, state_size):
        # possible actions of the agent
        self.ACTIONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'WAIT']
        self.state_size = state_size

    def convert_states_to_features(self, old_state, new_state, logger):
        if old_state is None:
            return None, None

        return self.convert_state_to_features(old_state, logger), \
            self.convert_state_to_features(new_state, logger)

    def convert_state_to_features(self, state, logger):
        # just want the direction to the nearest coin and the surrounding moving options as features
        # [CoinUpDown][CoinLeftRight][LeftWallNoWall][RightWallNoWall][UpWallNoWall][DownWallNoWall]
        encoded_state = np.array([])
        pos_agent = (state['self'])[3]
        coins = state['coins']
        field = state['field']

        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_coin(pos_agent, coins, field))
        encoded_state = np.append(encoded_state, self.calc_surrounding_features(pos_agent, field))

        return encoded_state

    def perform_qlearning(self, features_old, features_new, action, reward):
        #convert the state into to featurespace of our model
        index_old = self.state_to_index(features_old)
        index_new = self.state_to_index(features_new)
        action_index = self.action_to_index(action)
        #performe the q-learning process
        followup_reward = np.amax(self.Q_TABLE[index_new])
        self.Q_TABLE[index_old][action_index] = (1 - self.ALPHA) * self.Q_TABLE[index_old][action_index] + \
            self.ALPHA * (reward + self.GAMMA * followup_reward)
        #updating the hyper-values of training
        self.HOR_DISTANCE = features_new[0]
        self.VER_DISTANCE = features_new[1]
        self.ROUTE_PLANER = features_new[2]
    @staticmethod
    def calc_distance_and_direction_coin(pos_agent, coins, field):
        # picking some very long distance
        best_distance = 1000.0
        hor_distance = 1000.0
        ver_distance = 1000.0
        closest_coin = None
        #if there is no more coin on the board, return zeros
        if len(coins) == 0:
            return np.zeros((3))
        else:
            for coin in coins:
                '''
                calculating the horizontal and vertical distance between the coin and the agent
                the distance is the sum of horizontal and vertical distance, since the agent cant move diagonal
                '''
                horizontal_distance = np.abs(coin[0]-pos_agent[0])
                vertical_distance = np.abs(coin[1]-pos_agent[1])
                distance_coin = horizontal_distance + vertical_distance
                surroundings = CoinRlModel.calc_surrounding_features(pos_agent, field)
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
            #print("closest coin:" +str(closest_coin)+"distance:"+str(best_distance))
            #print("horizontal_distance:"+str(hor_distance),"vertical_distance:"+str(ver_distance))
            return feature

    @staticmethod
    def calc_surrounding_features(pos_agent, field):
        """
        Calculatiing the surrounding features far a given state

        feature[0] = left
        feature[1] = right
        feature[2] = down
        feature[3] = up
        """
        feature = np.zeros((4))
        x = pos_agent[0]
        y = pos_agent[1]
        feature[0] = field[x - 1][y]
        feature[1] = field[x + 1][y]
        feature[2] = field[x][y + 1]
        feature[3] = field[x][y - 1]
        return feature

    def calc_rewards(self, events):
        """
        Calculating rewards:
        """

        """
        Negative reward for each move to minimize the game-time needed
        """
        rewards = -50
        for event in events:
            if event == e.COIN_COLLECTED:
                rewards += 1000
                continue

            if event in [e.MOVED_UP, e.MOVED_DOWN, e.MOVED_LEFT, e.MOVED_RIGHT]:
                """
                Rewarding the move commands:
                
                reducing the distance to the nearest coin is rewarded positively
                while increasing the distance is rewarded negatively
                
                reducing the distance in the direction of biggest gradient is rewarded
                more positively since it decreases the chance of getting blocked by a boulder
                """
                if (event == e.MOVED_UP):
                    if (int(self.VER_DISTANCE) == 1):
                        rewards += 150
                        if (int(self.ROUTE_PLANER) != 1):
                            rewards += 150
                    if (int(self.VER_DISTANCE) == -1):
                        rewards -= 150
                    continue
                if (event == e.MOVED_DOWN):
                    if (int(self.VER_DISTANCE) == -1):
                        rewards += 150
                        if (int(self.ROUTE_PLANER) != 1):
                            rewards += 150
                    if (int(self.VER_DISTANCE) == 1):
                        rewards -= 150
                    continue
                if (event == e.MOVED_LEFT):
                    if (int(self.HOR_DISTANCE) == 1):
                        rewards += 150
                        if (int(self.ROUTE_PLANER) != -1):
                            rewards += 150
                    if (int(self.HOR_DISTANCE) == -1):
                        rewards -= 150
                    continue
                if (event == e.MOVED_RIGHT):
                    if (int(self.HOR_DISTANCE) == -1):
                        rewards += 150
                        if (int(self.ROUTE_PLANER) != -1):
                            rewards += 150
                    if (int(self.HOR_DISTANCE) == 1):
                        rewards -= 150
                    continue
            """
            Waiting and invalid action are rewarded strongly negative, since they do not help to win the game
            """
            if event in [e.WAITED]:
                rewards -= 500
            if event == e.INVALID_ACTION:
                rewards -= 1000
        """
        Some debug stuff
        """
        #print("HOR_DISTANCE" +str(self.HOR_DISTANCE),"VER_DISTANCE" +str(self.VER_DISTANCE))
        #print("Route:" + str(self.ROUTE_PLANER))
        #print("Rewards" +str(rewards))
        return rewards

    def create_qtable(self):
        """
        creating a q-table of need size

        here we need 2187 rows, since there are 6 features which all can hold 3 different values
            -> 3^6 = 2187

        and we need 5 columns, since there are 5 possible actions in this scenario
        """
        self.Q_TABLE = np.zeros((2187, 5))
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

        Each state gets a ternary numeral value (Dreierzahlen-system) as index
        """
        temp = np.dot((state + 1), np.array([729, 243, 81, 27, 9, 3, 1]))
        return int(np.sum(temp, axis=0))

    def action_to_index(self, action):
        return int(np.where(np.array(self.ACTIONS) == action)[0][0])
