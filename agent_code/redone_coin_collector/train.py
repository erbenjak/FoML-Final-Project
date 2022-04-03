# this is training file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
from .redone_collector_model import CoinRlModelRedone
import os.path
import numpy as np

abs_path = "../learningProgress/redone_coin_qtable.npy"

def setup_training(self):
    """
        * called at begin of game
        * pass training variables into the self-object
        * the total reward is stored
    """
    self.model = CoinRlModelRedone(self.logger, 2187, abs_path)
    self.model.setup()
    self.results = np.array([])


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
        * analyze what has happened in the game and calculate the reward from this
    """
    if old_game_state is None:
        return
    self.model.performLearning(old_game_state, new_game_state, self_action, events)


def end_of_round(self, last_game_state, last_action, events):
    self.results = np.append(self.results, last_game_state['self'][1])
    np.save('score_tracker.npy', self.results)

    self.model.store_current_qtbale()
    return 1