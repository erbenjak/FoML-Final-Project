import os
import pickle
import random
import numpy as np
from collections import deque
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'sarsa_agent/sarsa_model.pt')


def setup(self):

    if self.train:
        self.logger.debug('Successfully entered setup code for training with rule based agent')
        np.random.seed()
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)

        self.ignore_others_timer = 0
        self.current_round = 0

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


def act(self, game_state: dict) -> str:

    if self.train:

        return act_rule_based(self, game_state)

    self.logger.debug("Querying model for action.")
    x = state_to_features(game_state)
    response = np.ravel([model.predict([x.ravel()]) for model in self.model])

    # to avoid loops: if the largest and second largest response differ only by 0.05, choose the second
    # best action with probability 50%
    if np.abs(np.sort(response)[-1] - np.sort(response)[-2]) < 0.05:
        index = np.random.choice(np.argsort(response)[-2:])
        return ACTIONS[index]

    self.logger.debug(f'Model chose action {ACTIONS[np.argmax(response)]}')

    return ACTIONS[np.argmax(response)]


def state_to_features(game_state: dict) -> np.array:

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field = game_state['field']
    coins = game_state['coins']
    position_x, position_y = game_state['self'][3]
    distance = np.zeros((len(coins), 2))
    i = 0
    for (x, y) in coins:
        dist_x = position_x - x
        dist_y = position_y - y
        distance[i, 0] = dist_x
        distance[i, 1] = dist_y
        i += 1
    assert len(np.sum(distance, axis=1)) == len(coins)
    try:
        features = distance[np.argmin(np.sum(distance ** 2, axis=1))]

    except ValueError:
        distance = np.array([np.random.choice([100, -100]), np.random.choice([100, -100])])
        features = distance
    assert len(features) == 2

    environment = np.zeros(4)
    if field[position_x - 1, position_y] == 0:
        environment[0] = 1
    if field[position_x + 1, position_y] == 0:
        environment[1] = 1
    if field[position_x, position_y - 1] == 0:
        environment[2] = 1
    if field[position_x, position_y + 1] == 0:
        environment[3] = 1

    features = np.append(features, environment)

    return features


# -------------------------------------------------------------------------
# code for the rule based agent: (act_rule_based is a modified version of rule_based_agent/callbacks/act)


def look_for_targets(free_space, start, targets, logger=None):

    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)

        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')

    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)

    self.ignore_others_timer = 0


def act_rule_based(self, game_state: dict) -> str:

    # FOR GENERATING TRAINING DATA ONLY
    # Exploration:
    # choose a random action with probability 70%
    random_prob = .7
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .15, .05])

    # ---------------------------------------

    self.logger.info('Picking action according to rule set')

    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')

    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates

    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    if (x, y) in dead_ends:
        action_ideas.append('BOMB')

    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')

    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')

            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):

            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')

    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a
