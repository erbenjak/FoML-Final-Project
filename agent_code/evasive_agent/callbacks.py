# this is execution file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
from .evasive_model import EvasiveModel
import random

abs_path_q_table = "evasive_agent_qtable.npy"
abs_path_seen_sates_table = "evasive_agent_seen_representations.npy"

def setup(self):
    self.model = EvasiveModel(self.logger, 531441, abs_path_q_table, abs_path_seen_sates_table)
    self.model.setup()

def act(self, game_state: dict):
    return self.model.playGame(self.train, game_state)