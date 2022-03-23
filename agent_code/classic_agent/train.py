# this is training file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
from .classic_model import ClassicModel
import os.path
import numpy as np

abs_path_q_table = "classic_agent_qtable.npy"
abs_path_seen_sates_table = "classic_agent_seen_representations.npy"

def setup_training(self):
    """
        * called at begin of game
        * pass training variables into the self-object
        * the total reward is stored
    """
    self.model = ClassicModel(self.logger, 531441, abs_path_q_table, abs_path_seen_sates_table)
    self.model.setup()


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
        * analyze what has happened in the game and calculate the reward from this
    """
    if old_game_state is None:
        return
    self.model.performLearning(old_game_state, new_game_state, self_action, events)


def end_of_round(self, last_game_state, last_action, events):
    self.model.performEndOfGameLearning(last_game_state, last_action, events)
    self.model.store_current_qtbale(abs_path_q_table,abs_path_seen_sates_table)
    return 1