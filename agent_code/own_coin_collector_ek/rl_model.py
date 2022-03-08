import events as e
import numpy as np


class CoinRlModel:
    """
    This class represents the regression model and will therefor keep track of its current features and
    will be calculating the rewards for when a event occurs
    """
    ACTIONS = []
    Q_TABLE = np.array([])
    state_size = -1
    # defining the hyper-parameters
    ALPHA = 0.1
    GAMMA = 0.6
    EPSILON = 1.0
    LAST_MOVE = 'WAIT'

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
        self.Q_TABLE[index_old][action_index] = (1- self.ALPHA) * self.Q_TABLE[index_old][action_index] + \
            self.ALPHA * (reward + self.GAMMA * followup_reward)
        self.LAST_MOVE = action
        return

    @staticmethod
    def calc_direction_coin(pos_agent, coins):
        # picking some very long distance
        best_distance = 1000.0
        closest_coin = None

        for coin in coins:
            distance_coin = (coin[0]-pos_agent[0]) ^ 2 + (coin[1]-pos_agent[1]) ^ 2
            if distance_coin < best_distance:
                best_distance = distance_coin
                closest_coin = coin

        feature = np.zeros((2))

        if pos_agent[0] < closest_coin[0]:
            feature[0] = -1
        elif pos_agent[0] > closest_coin[0]:
            feature[0] = 1

        if pos_agent[0] < closest_coin[0]:
            feature[1] = -1
        elif pos_agent[0] > closest_coin[0]:
            feature[1] = 1

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
        rewards = 0
        for event in events:
            if event == e.COIN_COLLECTED:
                rewards += 100
                continue

            if event in [e.MOVED_UP, e.MOVED_DOWN, e.MOVED_LEFT, e.MOVED_RIGHT]:
                if (event == e.MOVED_UP) & (self.LAST_MOVE == 'DOWN'):
                    rewards -= 50
                    continue
                if (event == e.MOVED_DOWN) & (self.LAST_MOVE == 'UP'):
                    rewards -= 50
                    continue
                if (event == e.MOVED_LEFT) & (self.LAST_MOVE == 'RIGHT'):
                    rewards -= 50
                    continue
                if (event == e.MOVED_RIGHT) & (self.LAST_MOVE == 'LEFT'):
                    rewards -= 50
                    continue

            if event in [e.WAITED]:
                rewards -= 50
            if event == e.INVALID_ACTION:
                rewards -= 75
        return rewards

    def create_qtable(self):
        self.Q_TABLE = np.zeros((729, 5))
        return

    def get_qtable(self):
        return self.Q_TABLE

    def set_qtable(self, qtable):
        self.Q_TABLE = qtable
        return

    @staticmethod
    def state_to_index(state):
        temp = (state + 1) * np.array([243, 81, 27, 9, 3, 1])
        return int(np.sum(temp, axis=0))

    def action_to_index(self, action):
        return int(np.where(np.array(self.ACTIONS) == action)[0][0])