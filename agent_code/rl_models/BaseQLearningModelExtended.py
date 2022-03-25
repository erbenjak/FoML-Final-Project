import numpy as np
import os.path
from .BaseModel import BaseModel
import random
import math
import time
"""
    Manages the Basic Q-Learning processes while leaving the feature and reward calculations to the concreate agent
    implementations.  
"""
class BaseQLearningModel(BaseModel):
    NUMBER_FEATURES = 18
    NUMBER_POSSIBLE_MOVES = 6

    """
    The q-table changes a whole lot in this iteration for one we are not storing all states inside it, but are only
    keeping track of the states, that actually appeared.
    """
    # the q table is now to be understood as a dictionary
    # as each state can be converted into a single number.
    # now we at what array index of the state is stored if
    # it is stored we can get its internal index and find with it the action values
    SEEN_FEATURES = np.ndarray(shape=(0, NUMBER_FEATURES))
    Q_VALUES = np.ndarray(shape=(0, NUMBER_POSSIBLE_MOVES))

    ALPHA = 0.10
    GAMMA = 0.9
    EPSILON_THRESHOLD = 0.05
    EPSILON_START = 1.1
    EPSILON_DECAY_PARAM_1 = 0.5
    EPSILON_DECAY_PARAM_2 = 0.2
    EPSILON_DECAY_PARAM_3 = 0.1
    NUM_ROUNDS_TRAINING = 10000
    DISTANCE_THRESHOLD = 6

    # the max_feature_size determines all possible feature-states
    max_feature_size = -1
    path_q_table = None
    path_seen_representations = None

    def __init__(self, logger, max_feature_size, path_q_tabel, path_seen_representations,
                 alpha=0.15, gamma=0.10, epsilon=0.05):
        BaseModel.__init__(self, logger)
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.max_feature_size = max_feature_size
        self.path_q_table = path_q_tabel
        self.path_seen_representations = path_seen_representations

    # -------------------setup----------------------------------------------------
    def setup(self):
        self.logger.info("Setting up the q-learning agent")

        if not os.path.isfile(self.path_q_table):
            self.create_qtable()
        else:
            self.load_current_qtable(self.path_q_table, self.path_seen_representations)

    # ------------------progress--------------------------------------------------
    def store_progress(self):
        self.logger.info("Storing learning progress")
        self.store_current_q_table(self.path_q_table, self.path_seen_representations)

    # ------------------pass on headers-------------------------------------------
    def calculateReward(self, events):
        """
        Calculate the reward an agent receives for a certain events.
        The return may be any float.
        """
        raise NotImplementedError("A model must provide a methode to calculate it's reward. ")

    def calculateFeaturesFromState(self, state):
        """
        Takes a game-state and extracts the features which are required to take good decisions.
        """
        raise NotImplementedError("A model must provide a methode to turn a game-state into features. ")

    @staticmethod
    def state_to_index(state):
        """
        The index calculation depends upon the features chosen and the range of values that these features can take on
        """

    def compute_additional_rewards(self, events, new_state, old_state):
        """
        Takes a set of events produced by the game engine and adds some custom events to be able to
        add some additional self-defined 'custom' events
        """
        raise NotImplementedError("A model must be able to calculate some additional events to make custom "
                                  "rewards possible.")

    # ---------------- managing the q-learning properties --------------------------

    def performLearningSingleState(self, single_state_old, single_state_new, single_action, reward):
        """
        Perform the learning for a single instances. (Meaning a version after mirroring and rotating)
        -----
        params:
        single_state_old: previously occurred state
        single_state_new: state which was reached starting from single_state_old using the given action,
        action: action used to get from the old to the new state
        reward: the gained during this progress
        -----
        returns: None
        """

        # convert the state into to feature-space of our model
        features_old = self.calculateFeaturesFromState(single_state_old)
        features_new = self.calculateFeaturesFromState(single_state_new)

        feature_q_table_old = np.where((self.SEEN_FEATURES == features_old).all(axis=1))
        if len(feature_q_table_old[0]) > 1:
            self.logger.info("FATAL ERROR OLD FEATURE IS STORED " + str(len(feature_q_table_old[0])) + " TIMES")
            self.logger.info(features_old)
            self.logger.info(single_action)
            raise ValueError('q table is in an illegal state')

        if len(feature_q_table_old[0]) == 0:
            index_q_table_old = int(self.add_new_state(features_old))
        else:
            index_q_table_old = feature_q_table_old[0][0]

        feature_q_table_new = np.where((self.SEEN_FEATURES == features_new).all(axis=1))

        if len(feature_q_table_new[0]) > 1:
            self.logger.info("FATAL ERROR NEW FEATURE IS STORED " + str(len(feature_q_table_new[0])) + " TIMES")
            self.logger.info(features_new)
            self.logger.info(single_action)
            raise ValueError('q table is in an illegal state')

        if len(feature_q_table_new[0]) == 0:
            index_q_table_new = self.add_new_state(features_new)
        else:
            index_q_table_new = feature_q_table_new[0][0]

        action_index = self.action_to_index(single_action)

        # perform the q-learning process - update the q-table
        followup_reward = np.amax(self.Q_VALUES[int(index_q_table_new)])
        self.Q_VALUES[index_q_table_old][action_index] = \
            (1 - self.ALPHA) * self.Q_VALUES[index_q_table_old][action_index] + \
            self.ALPHA * (reward + self.GAMMA * followup_reward)

    def add_new_state(self, feature_representation):
        """
        Adds a new state to the SEEN_REPRESENTATIONS and this states initial values to the Q_VALUES
        -----
        params:
        feature_representation: state to added in featured form
        -----
        returns: index of the added state
        """
        # find_closest state
        move_values, distance = self.find_closest_guess(feature_representation)

        if distance > self.DISTANCE_THRESHOLD:
            # get educated guess
            self.logger.info("educated guess provides the initial values")
            move_values = self.build_educated_initial_guess(feature_representation)
        else:
            self.logger.info("nearest -neighbour provides the initial values!")

        # store new initial result
        self.SEEN_FEATURES = np.append(self.SEEN_FEATURES, np.expand_dims(feature_representation, axis=0), axis=0)
        self.Q_VALUES = np.append(self.Q_VALUES, np.expand_dims(move_values, axis=0), axis=0)

        return (self.Q_VALUES.size / self.NUMBER_POSSIBLE_MOVES) - 1

    def performLearningLastState(self, last_state, last_action, reward):
        """
        Perform the learning for the last game state.
        """
        followup_reward = 0

        features = self.calculateFeaturesFromState(last_state)
        index_possibilities = np.where((self.SEEN_FEATURES == features).all(axis=1))

        if (len(index_possibilities[0]) != 1):
            self.logger.info("FATAL ERROR FEATURE IS STORED " + str(len(index_possibilities[0])) + " TIMES")
            self.logger.info(features)
            raise ValueError('q table is in an illegal state')

        index = int(index_possibilities[0][0])
        action_index = self.action_to_index(last_action)

        self.Q_VALUES[index][action_index] = (1 - self.ALPHA) * self.Q_VALUES[index][action_index] + \
                                             self.ALPHA * (reward + self.GAMMA * followup_reward)

    def get_epsilon(self, round):
        # here an epsilon greedy policy is required
        # for now we assume, that 1000 games will be played - hard coded
        # furthermore we assume, that we want to start of at about 1 and end at about 5%

        return 0.1

        standardized_round = (round - self.EPSILON_DECAY_PARAM_1 * self.NUM_ROUNDS_TRAINING) / \
                             (self.EPSILON_DECAY_PARAM_2 * self.NUM_ROUNDS_TRAINING)
        cosh = np.cosh(math.exp(-standardized_round))
        epsilon = 1.1 - (1 / cosh + (round * self.EPSILON_DECAY_PARAM_3 / self.NUM_ROUNDS_TRAINING))

        if epsilon < self.EPSILON_THRESHOLD:
            return self.EPSILON_THRESHOLD

        return epsilon

    ################ playing the actual game with the given information ###########
    def playGame(self, train, state):
        feature_representation = self.calculateFeaturesFromState(state)

        """plays the game using the trained qtable"""

        # this will not work if the SEEN_FEATURES array is empty
        new_state_encountered = True

        if self.SEEN_FEATURES.size != 0:
            index_q_table = np.where((self.SEEN_FEATURES == feature_representation).all(axis=1))
            if (len(index_q_table[0]) > 1):
                self.logger.info("FATAL ERROR NEW INDEX IS STORED " + str(len(index_q_table[0])) + " TIMES")
                raise ValueError('q table is in an illegal state')

            if len(index_q_table[0]) != 0:
                new_state_encountered = False

        move_values = None

        if new_state_encountered is True:
            allGameStates = self.multiply_game_state(state)

            for gameState in allGameStates:
                single_feature_representation = self.calculateFeaturesFromState(gameState)
                # if this feature is already present maybe because a mirrored or rotated version is equal to another
                # it does not need to be added
                index_representation_table = np.where((self.SEEN_FEATURES == single_feature_representation).all(axis=1))
                if len(index_representation_table[0]) > 0:
                    continue

                # find_closest state
                move_values_single_state, distance = self.find_closest_guess(single_feature_representation)

                if distance > self.DISTANCE_THRESHOLD:
                    # get educated guess
                    move_values_single_state = self.build_educated_initial_guess(single_feature_representation)

                # store new initial result
                self.SEEN_FEATURES = np.append(self.SEEN_FEATURES,
                                               np.expand_dims(single_feature_representation, axis=0), axis=0)
                self.Q_VALUES = np.append(self.Q_VALUES, np.expand_dims(move_values_single_state, axis=0), axis=0)

                if move_values is None:
                    move_values = move_values_single_state
        else:
            index_q_table = index_q_table[0][0]
            move_values = self.Q_VALUES[int(index_q_table)]

        move = np.argmax(move_values)
        move_val = move_values[move]

        if np.where(move_values == move_val)[0].shape[0] > 1:
            possible_move_indices = np.where(move_values == move_val)[0]
            # self.logger.info("Choosing from the following values: " + str(possible_move_indices))
            np.random.shuffle(possible_move_indices)
            move = possible_move_indices[0]

        chosen_action = self.ACTIONS[int(move)]
        if train:
            if random.random() < self.get_epsilon(state['round']):
                action_chosen = self.getActions()[int(random.randint(0, 5))]
                self.add_move_to_memory(feature_representation, action_chosen)
                chosen_action = action_chosen

        # prevent getting stuck:
        action = self.prevent_getting_stuck(state)
        if action is not None:
            chosen_action = action

        self.add_move_to_memory(feature_representation, chosen_action)
        return chosen_action

    def find_closest_guess(self, feature_representation):
        if self.SEEN_FEATURES.size == 0:
            return None, self.DISTANCE_THRESHOLD + 1

        weights = np.ndarray(shape=feature_representation.shape)
        weights = np.ones(shape=feature_representation.shape)

        # just pick the first with the smallest distance
        differences = (self.SEEN_FEATURES[:] != feature_representation)*weights
        #self.logger.info('Single Distance: SEEN,NEW,DISTANCE:' + str(self.SEEN_FEATURES[0]) + str(feature_representation) + str(differences[0]))
        differences_final = np.sum(np.square(differences), axis=1)
        #self.logger.info("Distance" + str(differences_final))
        #just pick the first with the smallest distance
        index_smallest = np.argmin(differences_final)
        return self.Q_VALUES[index_smallest], differences_final[index_smallest]

    def build_educated_initial_guess(self, state):
        return np.zeros(6)

    def prevent_getting_stuck(self, state):
        """
        We want to improve training. In these cases we are aware that the same
        will pop up continuously only to be broken up by as to special cases in order to improve our n exploration. In
        order to prevent our model from over-learning certain states. We allow the agent to then take a
        probabilistic approach to the next step. Allowing to carry on.
        """
        feature = self.calculateFeaturesFromState(state)
        index_possibilities = np.where((self.SEEN_FEATURES == feature).all(axis=1))

        if len(index_possibilities[0]) != 1:
            self.logger.info("FATAL ERROR FEATURE IS STORED " + str(len(index_possibilities[0])) + " TIMES")
            self.logger.info(feature)
            raise ValueError('q table is in an illegal state')

        feature_ind = index_possibilities[0][0]

        short_term_memory = self.memory_short.copy()
        if len(short_term_memory) > 4:
            hist1 = short_term_memory.pop()
            hist2 = short_term_memory.pop()
            hist3 = short_term_memory.pop()
            hist4 = short_term_memory.pop()

            back_and_forth = (hist1[1] == hist3[1] and hist2[1] == hist4[1] and hist1[1] != hist2[1])
            waiting = (np.all(hist1[0] == hist2[0]) and hist1[1] == hist2[1] and hist1[1] == 'WAIT')
            if back_and_forth or waiting:
                current_table_state = self.Q_VALUES[feature_ind]

                # if the current state only contains 0s then we would divide by 0
                if np.all(current_table_state == 0):
                    current_table_state = current_table_state + 1
                if np.any(current_table_state < 0):
                    current_table_state = current_table_state + (2 * abs(np.min(current_table_state)))

                # preventing the dropping of random bombs
                current_table_state[5] = 0

                # probabilities are determined by dividing through the sum
                probabilities = current_table_state / np.sum(current_table_state)
                action = np.random.choice(self.ACTIONS, p=probabilities)
                return action

        return None

    # ------------ storing and loading the q-table ----------------------
    def load_current_qtable(self, path_q_table, path_seen_rep):
        self.logger.info("loading Q-Table")
        self.Q_VALUES = np.load(path_q_table)
        self.SEEN_FEATURES = np.load(path_seen_rep)
        self.logger.info("loaded Q-Table")

    def create_qtable(self):
        self.Q_VALUES = np.ndarray(shape=(0, self.NUMBER_POSSIBLE_MOVES))
        self.SEEN_FEATURES = np.ndarray(shape=(0, self.NUMBER_FEATURES))

    def store_current_q_table(self, path_q_table, path_seen_rep):
        if self.Q_VALUES is not None:
            self.logger.info("Saving current state of the q-table")
            np.save(path_q_table, self.Q_VALUES)
            np.save(path_seen_rep, self.SEEN_FEATURES)
