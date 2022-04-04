import pickle
from collections import namedtuple, deque
from typing import List
from sklearn.ensemble import GradientBoostingRegressor
import os
import numpy as np


import events as e
from .callbacks import state_to_features

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_action'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_INDEX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'sarsa_agent/sarsa_model.pt')

GAMMA = 0.95
FEATURE_HISTORY_SIZE = 10000




def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transition = {'state': None, "action": None, "next_state": None, "reward": None, "next_action": None}
    self.x = [deque(maxlen=FEATURE_HISTORY_SIZE) for _ in ACTIONS]  # features
    self.y = [deque(maxlen=FEATURE_HISTORY_SIZE) for _ in ACTIONS]  # targets
    if not self.model:  # initialise model
        print("initialising model")

        self.model = [
            GradientBoostingRegressor(n_estimators=1000, learning_rate=0.001, max_depth=1,
                                      random_state=0, loss='ls', warm_start=True, init='zero') for _ in ACTIONS]
        self.model_initialised = False
    else:
        self.model_initialised = True


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None and new_game_state is not None and self_action is not None:

        old_features = state_to_features(old_game_state)
        new_features = state_to_features(new_game_state)

        if new_features[0] ** 2 + new_features[1] ** 2 < old_features[0] ** 2 + old_features[1] ** 2:
            events.append(e.DECREASED_DISTANCE)
        if new_features[0] ** 2 + new_features[1] ** 2 > old_features[0] ** 2 + old_features[1] ** 2:
            events.append(e.INCREASED_DISTANCE)

        if self.transition["state"] is None:
            self.transition["state"] = old_features
            self.transition["action"] = self_action
            self.transition["next_state"] = new_features
            self.transition["reward"] = reward_from_events(self, events)
        else:
            self.transition["next_action"] = self_action

            index = ACTION_INDEX[self.transition["action"]]
            index_next = ACTION_INDEX[self.transition["next_action"]]
            if not self.model_initialised:
                x = self.transition["state"]

                y = self.transition["reward"]
            else:
                x = self.transition["state"]

                y = self.transition["reward"] + \
                    GAMMA * self.model[index_next].predict([self.transition["next_state"].ravel()])[0]

            self.transition["state"] = old_features
            self.transition["action"] = self_action
            self.transition["next_state"] = new_features
            self.transition["reward"] = reward_from_events(self, events)

            self.x[index].append(x)
            self.y[index].append(y)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    if last_action is not None:
        if self.transition["next_action"] is None:
            x = self.transition["state"]
            y = self.transition["reward"]
            index = ACTION_INDEX[self.transition["action"]]
            self.x[index].append(x)
            self.y[index].append(y)
        elif self.transition["next_action"] is not None:
            self.transition["next_action"] = last_action

            index = ACTION_INDEX[self.transition["action"]]
            index_next = ACTION_INDEX[self.transition["next_action"]]
            if not self.model_initialised:
                x1 = self.transition["state"]
                y1 = self.transition["reward"]

                x2 = state_to_features(last_game_state)
                y2 = reward_from_events(self, events) + 0
            else:
                x1 = self.transition["state"]

                y1 = self.transition["reward"] + \
                     GAMMA * self.model[index_next].predict([self.transition["next_state"].ravel()])[0]
                x2 = state_to_features(last_game_state)

                y2 = reward_from_events(self, events) + 0

            self.x[index].append(x1)
            self.y[index].append(y1)
            self.x[index_next].append(x2)
            self.y[index_next].append(y2)


    if all([(len(x) == FEATURE_HISTORY_SIZE) for x in self.x]):
        print("Fitting model")
        for i, action in enumerate(ACTIONS):
            self.model[i].fit(self.x[i], self.y[i])
            self.x[i].clear()
            self.y[i].clear()
        self.model_initialised = True

        with open(MODEL_PATH, "wb") as file:
            pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 3,
        # e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -3,
        e.WAITED: -0.2,
        e.DECREASED_DISTANCE: 1,
        e.INCREASED_DISTANCE: -0.5,
        # e.GOT_KILLED: -5,
        e.KILLED_SELF: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
