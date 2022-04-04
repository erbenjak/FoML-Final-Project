import os
import pickle
import random
import numpy as np


ACTIONS = ['UP', 'DOWN', 'LEFT','RIGHT', 'WAIT', 'BOMB']
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'one_step_agent/one_step_model.pt')

random_prob = 0


def setup(self):

    if self.train:
        if os.path.isfile(MODEL_PATH):
            self.logger.info("Loading model from saved state.")
            with open(MODEL_PATH, "rb") as file:
                self.model = pickle.load(file)
        else:
            self.logger.info("Setting up model from scratch.")
            self.model = None
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)
        self.logger.info('model loaded')


def act(self, game_state: dict) -> str:

    currentRandomProb = -1  # in- or decrease depending on mode
    if self.train and random.random() < currentRandomProb:
        self.logger.debug("Choosing random action.")
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    x = state_to_features(game_state)
    response = np.ravel([model.predict([x.ravel()]) for model in self.model])

    if np.abs(np.sort(response)[-1] - np.sort(response)[-2]) < 0.01:
        index = np.random.choice(np.argsort(response)[-2:])
        return ACTIONS[index]
    self.logger.debug(f'Model chose action {ACTIONS[np.argmax(response)]}')
    action = ACTIONS[np.argmax(response)]
    return action


def getSurroundingFields(field, position):

    """
    Get the tiles surrounding the agent
    """
    myX = position[0]
    myY = position[1]

    surrounding = []
    surrounding.append(field[myX - 1, myY - 1])
    surrounding.append(field[myX, myY - 1])
    surrounding.append(field[myX + 1, myY - 1])
    surrounding.append(field[myX - 1, myY])
    surrounding.append(field[myX + 1, myY])
    surrounding.append(field[myX - 1, myY + 1])
    surrounding.append(field[myX, myY + 1])
    surrounding.append(field[myX + 1, myY + 1])

    return surrounding


def state_to_features(game_state: dict) -> np.array:
    """
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # Dict before/after game
    if game_state is None:
        return None

    field = game_state['field']
    coins = game_state['coins']
    index_arrays = np.where(field == 1)
    crates = list(zip(index_arrays[0], index_arrays[1]))
    position_x, position_y = game_state['self'][3]
    if len(coins) > 0:
        distance = np.zeros((len(coins), 2))
        i = 0
        for (x, y) in coins:
            dist_x = position_x - x
            dist_y = position_y - y
            distance[i, 0] = dist_x
            distance[i, 1] = dist_y
            i += 1

        assert len(np.sum(distance, axis=1)) == len(coins)
        features = distance[np.argmin(np.sum(distance ** 2, axis=1))]  # (x,y) position relative to nearest coin
        # distance can be negative
        assert len(features) == 2
    else:
        distance = np.zeros((len(crates), 2))
        i = 0
        for (x, y) in crates:
            dist_x = position_x - x
            dist_y = position_y - y
            distance[i, 0] = dist_x
            distance[i, 1] = dist_y
            i += 1

        assert len(np.sum(distance, axis=1)) == len(crates)
        features = distance[np.argmin(np.sum(distance ** 2, axis=1))]  # distance in x and y direction to closest coin
        # distance can be negative
        assert len(features) == 2

    changedField = field
    for coin in game_state['coins']:
        changedField[coin] = 2
    for bomb in game_state['bombs']:
        bomb_coordinates = bomb[0]
        changedField[bomb_coordinates] = 3

    surrounding = getSurroundingFields(changedField, (position_x, position_y))

    for point in surrounding:
        features = np.append(features, point)

    return features
