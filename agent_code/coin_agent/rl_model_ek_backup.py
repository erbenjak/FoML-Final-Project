import events as e
import numpy as np
import random

class CoinRlModel:
    """
    This class represents the regression model and will therefor keep track of its current features and
    will be calculating the rewards for when a event occurs
    """
    ACTIONS = []
    Q_TABLE = np.array([])
    state_size = -1
    # defining the hyper-parameters
    ALPHA = 0.2
    GAMMA = 0.25
    EPSILON = 0.2
    LAST_MOVE = 'WAIT'
    # shortest distance to a coin
    VER_DISTANCE = 1000.0
    HOR_DISTANCE = 1000.0
    ROUTE_PLANER = 0
    #surrounding blocks
    BLOCKS=np.zeros((4))

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

        encoded_state = np.append(encoded_state, self.calc_direction_coin(pos_agent, coins))
        encoded_state = np.append(encoded_state, self.calc_surrounding_features(pos_agent, field))

        return encoded_state

    def perform_qlearning(self, features_old, features_new, action, reward):
        index_old = self.state_to_index(features_old)
        index_new = self.state_to_index(features_new)
        action_index = self.action_to_index(action)
        followup_reward = np.amax(self.Q_TABLE[index_new])
        self.Q_TABLE[index_old][action_index] = (1 - self.ALPHA) * self.Q_TABLE[index_old][action_index] + \
            self.ALPHA * (reward + self.GAMMA * followup_reward)
        #self.LAST_MOVE = action
        self.HOR_DISTANCE = features_new[0]
        self.VER_DISTANCE = features_new[1]
        self.ROUTE_PLANER = features_new[2]

        #self.BLOCKS = features_new[2:6]
    @staticmethod
    def calc_direction_coin(pos_agent, coins):
        # picking some very long distance
        best_distance = 1000.0
        closest_coin = None
        hor_distance = 1000.0
        ver_distance = 1000.0
        if len(coins) == 0:
            return np.zeros((3))
        else:
            for coin in coins:
                distance_coin = (np.abs(coin[0]-pos_agent[0]) + np.abs(coin[1]-pos_agent[1]))
                if distance_coin < best_distance:
                    best_distance = distance_coin
                    closest_coin = coin
                    hor_distance = np.abs(closest_coin[0]-pos_agent[0])
                    ver_distance = np.abs(closest_coin[1]-pos_agent[1])
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

            return feature

    @staticmethod
    def calc_surrounding_features(pos_agent, field):
        feature = np.zeros((4))
        x = pos_agent[0]
        y = pos_agent[1]
        feature[0] = field[x - 1][y]
        feature[1] = field[x + 1][y]
        feature[2] = field[x][y + 1]
        feature[3] = field[x][y - 1]
        return feature

    def calc_rewards(self, events):
        rewards = -100
        for event in events:
            if event == e.COIN_COLLECTED:
                rewards += 1000
                continue

            if event in [e.MOVED_UP, e.MOVED_DOWN, e.MOVED_LEFT, e.MOVED_RIGHT]:
                if (event == e.MOVED_UP):
                    if (int(self.VER_DISTANCE) == -1):
                        rewards += 150
                        if (int(self.ROUTE_PLANER) != 1):
                            rewards += 100
                    if (int(self.VER_DISTANCE) == 1):
                        rewards -= 100
                    continue
                if (event == e.MOVED_DOWN):
                    if (int(self.VER_DISTANCE) == 1):
                        rewards += 150
                        if (int(self.ROUTE_PLANER) != 1):
                            rewards += 100
                    if (int(self.VER_DISTANCE) == -1):
                        rewards -= 100
                    continue
                if (event == e.MOVED_LEFT):
                    if (int(self.HOR_DISTANCE) == 1):
                        rewards += 150
                        if (int(self.ROUTE_PLANER) != -1):
                            rewards += 100
                    if (int(self.HOR_DISTANCE) == -1):
                        rewards -= 100
                    continue
                if (event == e.MOVED_RIGHT):
                    if (int(self.HOR_DISTANCE) == -1):
                        rewards += 150
                        if (int(self.ROUTE_PLANER) != -1):
                            rewards += 100
                    if (int(self.HOR_DISTANCE) == 1):
                        rewards -= 100
                    continue
            if event in [e.WAITED]:
                rewards -= 500
            if event == e.INVALID_ACTION:
                rewards -= 1000
        #print(self.HOR_DISTANCE,self.VER_DISTANCE)
        #print(rewards)
        return rewards

    def create_qtable(self):
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
        temp = np.dot((state + 1),np.array([729,243, 81, 27, 9, 3, 1]))
        return int(np.sum(temp, axis=0))

    def action_to_index(self, action):
        return int(np.where(np.array(self.ACTIONS) == action)[0][0])