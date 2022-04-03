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
    self.results = np.array([])
    self.duration = np.array([])
    self.ratios = np.array([])
    self.quantiles = np.array([])

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
        * analyze what has happened in the game and calculate the reward from this
    """
    if old_game_state is None:
        return
    quan = self.model.performLearning(old_game_state, new_game_state, self_action, events)
    self.quantiles = np.append(self.quantiles,quan)

def end_of_round(self, last_game_state, last_action, events):
    self.results = np.append(self.results, last_game_state['self'][1])
    self.duration = np.append(self.duration, last_game_state['step'])
    self.ratios = np.append(self.ratios, self.model.getCounterPercentages())

    np.save("results.npy",self.results)
    np.save("duration.npy",self.duration)
    np.save("ratios.npy", self.ratios)
    np.save("quantiles.npy", self.quantiles)
    self.model.performEndOfGameLearning(last_game_state, last_action, events)
    self.model.store_current_q_table(abs_path_q_table,abs_path_seen_sates_table)
    self.model.resetCounter()
    return 1