# this is training file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
from .rl_model_ek import CoinRlModel
import os.path
import time
import numpy as np


def setup_training(self):
    """
        * called at begin of game
        * pass training variables into the self-object
        * the total reward is stored
    """
    self.reward_total = 0
    # as we have a 17 by 17 map
    self.model = CoinRlModel(289)
    # If it doesn't already exist we create our Q-Table here.
    if os.path.isfile('qtable.npy'):
        self.logger.info("Q-Table gets loaded")
        self.model.set_qtable(np.load('qtable.npy'))
    else:
        self.logger.info("Q-Table gets created")
        self.model.create_qtable()


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
        * analyze what has happened in the game and calculate the reward from this
    """
    reward = self.model.calc_rewards(events)
    vectorized_old, vectorized_new = self.model.convert_states_to_features(old_game_state, new_game_state, self.logger)
    if vectorized_old is None:
        return

    """
    some debug stuff
    """
    #print(old_game_state['self'][3])
    #print("step " + str(old_game_state['step']))
    #print("action:"+str(self_action))
    #print("reward:"+str(reward))
    #print(CoinRlModel.calc_surrounding_features(old_game_state['self'][3], old_game_state['field']))
    #time.sleep(5)
    self.model.perform_qlearning(vectorized_old, vectorized_new, self_action, reward)



def end_of_round(self, last_game_state, last_action, events):
    self.logger.info("Q-Table has been saved")
    q_table = self.model.get_qtable()
    np.save('qtable.npy', q_table)
    return 1
