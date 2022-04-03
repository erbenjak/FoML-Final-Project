# this is execution file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
from .redone_collector_model import CoinRlModelRedone
import random

abs_path = "../learningProgress/redone_coin_qtable.npy"

def setup(self):
    self.model = CoinRlModelRedone(self.logger, 2187, abs_path)
    self.model.setup()


def act(self, game_state: dict):
    if self.train:
        if random.random() < 0.051:
            return self.model.getActions()[int(random.randint(0, 5))]
    return self.model.playGame(self.train, game_state)

