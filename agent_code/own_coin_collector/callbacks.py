# this is execution file for the own_coin_collector-agents
# it's focus lies upon managing to perform the coin_heaven scenario efficiently
from .rl_model import CoinRlModel
import numpy as np
import os.path


def setup(self):
    self.model = CoinRlModel(289)
    # If it doesn't already exist we create our Q-Table here.
    if os.path.isfile('qtable.npy'):
        self.logger.info("Q-Table gets loaded")
        self.model.set_qtable(np.load('qtable.npy'))
    self.MOVES = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'WAIT']


def act(self, game_state: dict):
    #if train:
    #    return random_move

    converted_state = self.model.convert_state_to_features(game_state, self.logger)
    self.logger.info("decision function knows a encoded state:" + str(converted_state))
    q_index = self.model.state_to_index(converted_state)
    self.logger.info("decision function knows to inspect the following the q_index:" + str(q_index))
    move_values = self.model.get_qtable()[int(q_index)]
    self.logger.info("decision function knows the following move values" + str(move_values))
    move = np.argmax(move_values)

    # in case the highest value appears more than once choose a random action
    move_val = move_values[move]
    if np.where(move_values == move_val)[0].shape[0] > 1:
        possible_move_indecies = np.where(move_values == move_val)[0]
        self.logger.info("Choosing from the follwoing values: " + str(possible_move_indecies))
        np.random.shuffle(possible_move_indecies)
        move = possible_move_indecies[0]
    return self.MOVES[int(move)]
