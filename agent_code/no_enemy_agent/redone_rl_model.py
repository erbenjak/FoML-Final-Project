import events as e
import numpy as np
import random
from ..rl_models.BaseQLearningModel import BaseQLearningModel

class CoinRlModelRedone(BaseQLearningModel):
    """
    This is an implementation of the BaseQLearningModel.

    REMINDER: x- and y-axis of the board point right and down (matrix-view)
                they do NOT point right and up (typical physical view)
    """

    def __init__(self, logger, max_feature_size, path):
        BaseQLearningModel.__init__(self, logger, max_feature_size, path)

    ############### Necessary Overwrites ##############################################
    def compute_additional_rewards(self, events, new_state, old_state):
        """
        Takes a set of events produced by the game engine and adds some custom events to be able to
        add some additional self-defined 'custom' events
        """
        events = self.addCoinDistanceEvents(events,new_state,old_state)
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
        """
        Negative reward for each move to minimize the game-time needed
        """
        reward = 0

        for event in events:
            if event == e.COIN_COLLECTED:
                reward += 200
                continue
            if event in [e.MOVED_UP, e.MOVED_DOWN, e.MOVED_LEFT, e.MOVED_RIGHT]:
                reward -= 1
            if event in [e.WAITED]:
                reward -= 500
            if event == e.INVALID_ACTION:
                reward -= 1000
            if event == e.BOMB_DROPPED:
                reward -= 5000000000
            if event == 'DEC_COIN_DIST':
                self.logger.info("decreased coin distance event")
                reward += 20
            if event == "INC_COIN_DIST":
                self.logger.info("increased coin distance event")
                reward -= 20
        """
        Some debug stuff
        """
        # print("HOR_DISTANCE" +str(self.HOR_DISTANCE),"VER_DISTANCE" +str(self.VER_DISTANCE))
        # print("Route:" + str(self.ROUTE_PLANER))
        # print("Rewards" +str(rewards))
        return reward

    def calculateFeaturesFromState(self, state):
        """
        Takes a gamestate and extracts the features which are required to take good decisions.
        """
        # just want the direction to the nearest coin and the surrounding moving options as features
        # [CoinUpDown][CoinLeftRight][LeftWallNoWall][RightWallNoWall][UpWallNoWall][DownWallNoWall]
        encoded_state = np.array([])
        pos_agent = (state['self'])[3]
        coins = state['coins']
        field = state['field']
        bombs = state['bombs']
        encoded_state = np.append(encoded_state, self.calc_distance_and_direction_coin(pos_agent, coins, field))
        encoded_state = np.append(encoded_state, self.calc_surrounding_features(pos_agent, field, bombs))

        return encoded_state

    def calc_distance_and_direction_coin(self, pos_agent, coins, field):
            '''
            Simple encoding -1, 0, 1 for Left, no_x, Right
                            -1, 0, 1 for Down, no_y, Up
                            -1,0 for deadlock/no deadlock
            '''
            feature = np.zeros((3))
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


            # ToDo add deadlock information
            feature[2] = 0
            '''
            some debug stuff
            '''
            # print("closest coin:" +str(closest_coin)+"distance:"+str(best_distance))
            # print("horizontal_distance:"+str(hor_distance),"vertical_distance:"+str(ver_distance))
            return feature

    def calc_surrounding_features(self,pos_agent, field, bombs):
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
        explosions = np.zeros((4))
        for bomb in bombs:

            if int(bomb[0][0]) == int(x):
                if int(bomb[1]) == 0:
                    if np.abs(x - 1 - bomb[0][0]) == 0:
                        explosions[0] = 2
                    if np.abs(x + 1 - bomb[0][0]) == 0:
                        explosions[1] = 2
            if int(bomb[0][1]) == int(y):
                if int(bomb[1]) == 0:
                    if np.abs(y + 1 - bomb[0][1]) == 0:
                        explosions[2] = 2
                    if np.abs(y - 1 - bomb[0][1]) == 0:
                        explosions[3] = 2

        feature[0] = field[x - 1][y]
        feature[1] = field[x + 1][y]
        feature[2] = field[x][y + 1]
        feature[3] = field[x][y - 1]
        return feature

    def bomb_escaping_feature(self, x, y, bomb_pos,field, expl_map, others):
        """
        Check if a step allows the agent to escape an explosion.
        """


    @staticmethod
    def state_to_index(state):
        """
        Transforming the state variables into indices

        Each state gets a ternary numeral value (Dreierzahlen-system) as index
        """
        temp = np.dot((state + 1), np.array([1, 3, 9, 27, 108, 432, 1728]))
        return int(np.sum(temp, axis=0))