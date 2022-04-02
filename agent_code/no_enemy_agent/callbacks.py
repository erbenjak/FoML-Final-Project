# this is execution file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
import numpy as np
import events as e
import random

from .rl_model import RLModel
import os.path

abs_path = "qtable.npy"

def setup(self):
    self.model = RLModel(self.logger, 186624, abs_path)
    self.model.setup()


def act(self, game_state: dict):
    if self.train:
        if random.random() < self.model.get_epsilon():
            return self.model.getActions()[int(random.randint(0, 5))]
    return self.model.playGame(game_state)

