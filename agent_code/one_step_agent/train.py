import pickle
import os
from collections import namedtuple, deque
from typing import List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError

import numpy as np
import events as e
from .callbacks import state_to_features

# example
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_INDEX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'one_step_agent/one_step_model.pt')

from sklearn.utils.validation import check_is_fitted

# Hyper parameters
TRANSITION_HISTORY_SIZE = 3
RECORD_ENEMY_TRANSITIONS = 1.0
GAMMA = 0.95
POSITION_HISTORY_SIZE = 5

FEATURE_HISTORY_SIZE = 500


def setup_training(self):

    self.model_not_fitted = [False] * len(ACTIONS)
    self.position_history = deque(maxlen=POSITION_HISTORY_SIZE)
    self.x = [deque(maxlen=FEATURE_HISTORY_SIZE) for _ in ACTIONS]
    self.y = [deque(maxlen=FEATURE_HISTORY_SIZE) for _ in ACTIONS]
    if not self.model:
        print("initialising model")
        self.model = [
            GradientBoostingRegressor(n_estimators=1000, learning_rate=0.001, max_depth=1,

                                      random_state=0, loss='ls', warm_start=True, init='zero') for _ in ACTIONS]
        self.model_initialised = False
    else:
        self.model_initialised = True
        for index, model in enumerate(self.model):
            try:
                model.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            except NotFittedError:
                self.model_not_fitted[index] = True


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None and new_game_state is not None and self_action is not None:
        # state_to_features is defined in callbacks.py
        old_features = state_to_features(old_game_state)
        new_features = state_to_features(new_game_state)
        if new_features[0] ** 2 + new_features[1] ** 2 < old_features[0] ** 2 + old_features[1] ** 2:
            events.append(e.DECREASED_DISTANCE)
        if new_features[0] ** 2 + new_features[1] ** 2 > old_features[0] ** 2 + old_features[1] ** 2:
            events.append(e.INCREASED_DISTANCE)

        # check for loop and add reward if there is one
        current_position = new_game_state['self'][3]
        if len(self.position_history) > 4 and current_position == self.position_history[1] and current_position == \
                self.position_history[3]:
            events.append(e.STUCK_IN_LOOP)

        self.position_history.append(current_position)

        index = ACTION_INDEX[self_action]
        if not self.model_initialised or True in self.model_not_fitted:
            x = old_features
            y = reward_from_events(self, events)
        else:
            old_features = state_to_features(old_game_state)
            new_features = state_to_features(new_game_state)
            x = old_features
            x_new = new_features
            y = reward_from_events(self, events) + GAMMA * np.max(
                np.ravel([model.predict([x_new.ravel()]) for model in self.model]))

        self.x[index].append(x)
        self.y[index].append(y)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    if last_action is not None:
        index = ACTION_INDEX[last_action]
        if not self.model_initialised or True in self.model_not_fitted:
            x = state_to_features(last_game_state)
            y = reward_from_events(self, events)  # initial guess: Q = 0
        else:
            # SARSA
            old_features = state_to_features(last_game_state)
            x = old_features
            y = reward_from_events(self,
                                   events)

        self.x[index].append(x)
        self.y[index].append(y)

    for i, x in enumerate(self.x):
        print(len(x))
        if len(x) == FEATURE_HISTORY_SIZE:
            print("Fitting model")
            self.model[i].fit(self.x[i], self.y[i])
            self.x[i].clear()
            self.y[i].clear()
            self.model_not_fitted[i] = False
        self.model_initialised = True

    with open('steps.txt', 'a') as steps_log:
        steps_log.write(str(last_game_state['step']) + "\t")

    with open('scores.txt', 'a') as scores_log:
        scores_log.write(str(last_game_state['self'][1]) + "\t")

    # Store the model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.INVALID_ACTION: -3,
        e.DECREASED_DISTANCE: 1,
        e.INCREASED_DISTANCE: -0.5,  # to avoid loops?
        e.WAITED: -0.5,
        e.BOMB_DROPPED: 0.1,
        e.CRATE_DESTROYED: 5,
        e.KILLED_SELF: -1,
        e.MOVED_DOWN: -0.5,
        e.MOVED_UP: -0.5,
        e.MOVED_LEFT: -0.5,
        e.MOVED_RIGHT: -0.5,
        e.STUCK_IN_LOOP: -5,
    }
    reward_sum = 0
    for event in events:
        if event == 'STUCK_IN_LOOP':
            self.logger.info(f'loop occurred, negative reward applied')
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Sum of rewards {reward_sum} for events {', '.join(events)}")
    return reward_sum