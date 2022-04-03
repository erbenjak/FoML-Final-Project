# this is training file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
from .evasive_model import EvasiveModel
import os.path
import numpy as np

abs_path_q_table = "evasive_agent_qtable.npy"
abs_path_seen_sates_table = "evasive_agent_seen_representations.npy"

def setup_training(self):
    """
        * called at begin of game
        * pass training variables into the self-object
        * the total reward is stored
    """
    self.model = EvasiveModel(self.logger, 531441, abs_path_q_table, abs_path_seen_sates_table)
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
    self.model.performLearning(old_game_state, new_game_state, self_action, events)

def end_of_round(self, last_game_state, last_action, events):
    self.results = np.append(self.results, last_game_state['self'][1])
    self.duration = np.append(self.duration, last_game_state['step'])
    self.ratios = np.append(self.ratios, self.model.getCounterPercentages())
    q_values = self.model.getQValues()
    quan = np.zeros(5)
    quan[0] = np.quantile(q_values, 0)
    quan[1] = np.quantile(q_values, 0.25)
    quan[2] = np.quantile(q_values, 0.5)
    quan[3] = np.quantile(q_values, 0.75)
    quan[4] = np.quantile(q_values, 1)

    self.quantiles = np.append(self.quantiles, quan)
    np.save("results.npy",self.results)
    np.save("duration.npy",self.duration)
    np.save("ratios.npy", self.ratios)
    np.save("quantiles.npy", self.quantiles)

    self.model.performEndOfGameLearning(last_game_state, last_action, events)
    self.model.store_current_q_table(abs_path_q_table,abs_path_seen_sates_table)
    self.model.resetCounter()
    return 1