# this is training file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
from .rl_model import RLModel
import os.path
import numpy as np
import time

abs_path = "qtable.npy"

def setup_training(self):
    """
        * called at begin of game
        * pass training variables into the self-object
        * the total reward is stored
    """
    self.model = RLModel(self.logger, 186624, abs_path)
    self.model.setup()


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
        * analyze what has happened in the game and calculate the reward from this
    """
    if old_game_state is None:
        return
    self.model.performLearning(old_game_state, new_game_state, self_action, events)
    """
        some debug stuff
        """
    print(old_game_state['self'][3])
    print("step " + str(old_game_state['step']))
    print("action:"+str(self_action))

    print(self.model.calculateFeaturesFromState(old_game_state))
    print('events' + str(events))
    rewards_ = self.model.calculateReward(events)
    print('rewards' + str(rewards_))
    time.sleep(5)


def end_of_round(self, last_game_state, last_action, events):
    self.model.store_current_qtbale()
    return 1