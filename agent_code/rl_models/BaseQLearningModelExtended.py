import events as e
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

    NUMBER_FEATURES = 12
    NUMBER_POSSIBLE_MOVES = 6

    """
    The q-table changes a whole lot in this iteration for one we are not storring all states inside it, but are only
    keeping track of the states, that actually accured.
    """
    # the q table is now to be undestood as a dictionary
    # as each state can be converted into a single number as done bevore a
    # now we at what array index of the state is stored if
    # it is stored we can get it's internal index and find with it the action values
    SEEN_FEATURES = np.ndarray(shape=(0, NUMBER_FEATURES))
    Q_VALUES = np.ndarray(shape=(0,NUMBER_POSSIBLE_MOVES))


    ALPHA = 0.10
    GAMMA = 0.9
    EPSILON_TRESHHOLD = 0.05
    EPSILON_START = 1.1
    EPSILON_DECAY_PARAM_1 = 0.5
    EPSILON_DECAY_PARAM_2 = 0.2
    EPSILON_DECAY_PARAM_3 = 0.1
    NUM_ROUNDS_TRAINING = 10000
    DISTANCE_THRESHOLD = 6

    #the max_feature_size determines all possible featurized states
    max_feature_size = -1
    path_q_table = None
    path_seen_representations = None

    def __init__(self, logger, max_feature_size,path_q_tabel, path_seen_representations, ALPHA = 0.15, GAMMA = 0.10, EPSILON = 0.05):
        BaseModel.__init__(self,logger)
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.max_feature_size=max_feature_size
        self.path_q_table = path_q_tabel
        self.path_seen_representations = path_seen_representations

    ################## setup ####################################################
    def setup(self):
        self.logger.info("Setting up the q-learning agent")

        if os.path.isfile(self.path_q_table) == False:
            self.create_qtable()
        else:
            self.load_current_qtable(self.path_q_table,self.path_seen_representations)

    ################# progress ##################################################
    def store_progress(self):
        self.logger.info("Storing learning progress")
        self.store_current_qtbale(self.path_q_table,self.path_seen_representations)

    ################## pass on headers ##########################################
    def calculateReward(self, events):
        """
        Calculate the reward a agent receives for a certain events.
        The return may be any float.
        """
        raise NotImplementedError("A model must provide a methode to calculate it's reward. ")

    def calculateFeaturesFromState(self, state):
        """
        Takes a gamestate and extracts the features which are required to take good decisions.
        """
        raise NotImplementedError("A model must provide a methode to turn a gamestate into features. ")

    def state_to_index(state):
        """
        The index calculation depends upon the features choosen and the range of values that these features can take on
        """

    def compute_additional_rewards(self, events, new_state, old_state):
        """
        Takes a set of events produced by the game engine and adds some custom events to be able to
        add some additional self-defined 'custom' events
        """
        raise NotImplementedError("A model must be able to calculate some additional events to make custom "
                                  "rewards possible.")

    ################## managing the q-learning properties #######################
    def performLearningSingleState(self, singleStateOld, singleStateNew, singleAction, reward):
        """
        Perform the learning for a single instance.
        """
        # convert the state into to featurespace of our model
        features_old = self.calculateFeaturesFromState(singleStateOld)
        features_new = self.calculateFeaturesFromState(singleStateNew)

        feature_q_table_old = np.where((self.SEEN_FEATURES == features_old).all(axis=1))
        if(len(feature_q_table_old[0]) > 1):
            self.logger.info("FATEL ERROR OLD FEATURE IS STORRED " + str(len(feature_q_table_old[0])) + " TIMES")
            self.logger.info(features_old)
            self.logger.info(singleAction)
            raise ValueError('q table is in an illegal state')

        if(len(feature_q_table_old[0]) == 0):
            index_q_table_old = int(self.add_new_state(features_old))
        else:
            index_q_table_old = feature_q_table_old[0][0]

        feature_q_table_new = np.where((self.SEEN_FEATURES == features_new).all(axis=1))

        if(len(feature_q_table_new[0]) > 1):
            self.logger.info("FATEL ERROR NEW FEATURE IS STORRED " + str(len(feature_q_table_new[0])) + " TIMES")
            self.logger.info(features_new)
            self.logger.info(singleAction)
            raise ValueError('q table is in an illegal state')

        if (len(feature_q_table_new[0]) == 0):
            index_q_table_new = self.add_new_state(features_new)
        else:
            index_q_table_new = feature_q_table_new[0][0]

        action_index = self.action_to_index(singleAction)
        # perform the q-learning process
        followup_reward = np.amax(self.Q_VALUES[int(index_q_table_new)])
        self.Q_VALUES[index_q_table_old][action_index] = \
            (1 - self.ALPHA) * self.Q_VALUES[index_q_table_old][action_index] + \
            self.ALPHA * (reward + self.GAMMA * followup_reward)

        #self.logger.info("old state encountered: " + str(features_old))
        #self.logger.info("new state encountered: " + str(features_new))


    def add_new_state(self,feature_representation):
        self.logger.info("ADDING NEW STATE" + str(feature_representation))
        # find_closest state
        move_values, distance = self.find_clossest_guess(feature_representation)

        if distance > self.DISTANCE_THRESHOLD:
            # get educated guess
            self.logger.info("educated guess provides the initial values")
            move_values = self.build_educated_intial_guess(feature_representation)
        else:
            self.logger.info("nearest -neighbour provides the initial values!")

        # store new initial result
        #self.logger.info(self.SEEN_FEATURES)
        #self.logger.info(self.SEEN_FEATURES.shape)

        self.SEEN_FEATURES = np.append(self.SEEN_FEATURES, np.expand_dims(feature_representation, axis=0), axis=0)
        self.Q_VALUES = np.append(self.Q_VALUES, np.expand_dims(move_values, axis=0), axis=0)

        return (self.Q_VALUES.size / self.NUMBER_POSSIBLE_MOVES) - 1


    def performLearningLastState(self, lastState, lastAction, reward):
        """
        Perform the learning for the last game state.
        """
        followup_reward=0

        features = self.calculateFeaturesFromState(lastState)
        index_possibilities = np.where((self.SEEN_FEATURES == features).all(axis=1))

        if (len(index_possibilities[0]) != 1):
            self.logger.info("FATEL ERROR FEATURE IS STORRED " + str(len(index_possibilities[0])) + " TIMES")
            self.logger.info(features)
            raise ValueError('q table is in an illegal state')

        index = int(index_possibilities[0][0])
        action_index = self.action_to_index(lastAction)

        self.Q_VALUES[index][action_index] = (1 - self.ALPHA) * self.Q_VALUES[index][action_index] + \
                                                self.ALPHA * (reward + self.GAMMA * followup_reward)

    def get_epsilon(self, round):
        # here a epsilon greedy policy is required
        # for now we assume, that 1000 games will be played - hard coded
        # furthermore we assume, that we want to start of at about 1 and end at about 5%

        standardized_round = (round - self.EPSILON_DECAY_PARAM_1 * self.NUM_ROUNDS_TRAINING) / \
                             (self.EPSILON_DECAY_PARAM_2 * self.NUM_ROUNDS_TRAINING)
        cosh = np.cosh(math.exp(-standardized_round))
        epsilon = 1.1 - (1 / cosh + (round * self.EPSILON_DECAY_PARAM_3 / self.NUM_ROUNDS_TRAINING))

        if epsilon < self.EPSILON_TRESHHOLD:
            return self.EPSILON_TRESHHOLD

        return epsilon

    ################ playing the actual game with the given information ###########
    def playGame(self, train, state):
        feature_representation = self.calculateFeaturesFromState(state)

        """plays the game using the trained qtable"""

        # this will not work if the SEEN_FEATURES array is empty
        new_state_encountered = True

        if self.SEEN_FEATURES.size != 0:
            self.logger.info(self.SEEN_FEATURES)
            self.logger.info(feature_representation)
            index_q_table = np.where((self.SEEN_FEATURES == feature_representation).all(axis=1))
            if (len(index_q_table[0]) > 1):
                self.logger.info("FATEL ERROR NEW INDEX IS STORRED " + str(len(index_q_table[0])) + " TIMES")
                raise ValueError('q table is in an illegal state')

            if (len(index_q_table[0]) != 0):
                new_state_encountered = False

        move_values = None

        if new_state_encountered is True:
            self.logger.info("new state encountered ! " + str(feature_representation))
            #now one can go ahead and mirror and rotate the state
            allGameStates = self.multiply_game_state(state)

            for gameState in allGameStates:
                single_feature_representation = self.calculateFeaturesFromState(gameState)
                # if this feature is already present maybe because a mirrored or rotated version is equal to another
                # it does not need to be added
                index_representation_table = np.where((self.SEEN_FEATURES == single_feature_representation).all(axis=1))
                if (len(index_representation_table[0]) > 0):
                    continue

                self.logger.info("NEW STATE ADDED")
                # find_closest state
                move_values_single_state, distance = self.find_clossest_guess(single_feature_representation)

                if distance > self.DISTANCE_THRESHOLD:
                    # get educated guess
                    self.logger.info("educated guess provides the initial values")
                    move_values_single_state = self.build_educated_intial_guess(single_feature_representation)
                else:
                    self.logger.info("nearest -neighbour provides the initial values!")

                # store new initial result
                self.logger.info('this needs to be performed not once but for all mirroring and rotating')
                self.logger.info(self.SEEN_FEATURES)
                self.logger.info(self.SEEN_FEATURES.shape)

                self.SEEN_FEATURES = np.append(self.SEEN_FEATURES, np.expand_dims(single_feature_representation,axis=0),axis=0)
                self.Q_VALUES = np.append(self.Q_VALUES, np.expand_dims(move_values_single_state,axis=0), axis=0)

                if move_values is None:
                    move_values = move_values_single_state
        else:
            self.logger.info("known state encountered")
            index_q_table = index_q_table[0][0]
            move_values = self.Q_VALUES[int(index_q_table)]

        self.logger.info("decision function knows the following move values" + str(move_values))
        move = np.argmax(move_values)
        self.logger.info("decided for the following move : "+str(move))
        # in case the highest value appears more than once choose a random action
        move_val = move_values[move]
        if np.where(move_values == move_val)[0].shape[0] > 1:
            possible_move_indecies = np.where(move_values == move_val)[0]
            #self.logger.info("Choosing from the follwoing values: " + str(possible_move_indecies))
            np.random.shuffle(possible_move_indecies)
            move = possible_move_indecies[0]
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


    def find_clossest_guess(self,feature_representation):
        if (self.SEEN_FEATURES.size == 0):
            self.logger.info("debug" + str(self.SEEN_FEATURES.size))
            return None, self.DISTANCE_THRESHOLD+1

        weights = np.ndarray(shape=feature_representation.shape)
        weights = np.ones(shape=feature_representation.shape)

        differences = (self.SEEN_FEATURES[:] != feature_representation)*weights
        #self.logger.info('Single Distance: SEEN,NEW,DISTANCE:' + str(self.SEEN_FEATURES[0]) + str(feature_representation) + str(differences[0]))
        differences_final = np.sum(np.square(differences), axis=1)
        #self.logger.info("Distance" + str(differences_final))
        #just pick the first with the smallest distance
        index_smallest = np.argmin(differences_final)
        return self.Q_VALUES[index_smallest],differences_final[index_smallest]

    def build_educated_intial_guess(self,state):
        return np.zeros(6)

    def preventOverLearning(self, state):
        """
        We want to adrestraining. In these cases we are aware that the same
        will pop up continuously only to be broken up by as to special cases in order to improve our n exploration. In order to prevent our model from
        overlearning certain states. We allow the agent to then take a probabistic approach to the next step. Allowing
        us to live carry on.
        """
        feature = self.calculateFeaturesFromState(state)
        index_possibilities = np.where((self.SEEN_FEATURES == feature).all(axis=1))

        if (len(index_possibilities[0]) != 1):
            self.logger.info("FATEL ERROR FEATURE IS STORRED " + str(len(index_possibilities[0])) + " TIMES")
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
            waiting = (np.all(hist1[0] == hist2[0]) and hist1[1] == hist2[1] and hist1[1]=='WAIT')
            if back_and_forth or waiting:
                current_table_state = self.Q_VALUES[feature_ind]
                #self.logger.info(current_table_state)
                ### if the current state only contains 0s then we would divide by 0
                if(np.all(current_table_state==0)):
                    current_table_state = current_table_state + 1
                if np.any(current_table_state < 0):
                    current_table_state = current_table_state + (2 * abs(np.min(current_table_state)))
                ### preventing the dropping of random bombs
                #self.logger.info(current_table_state)
                current_table_state[5] = 0
                ### probabilities are determined by dividing through the sum
                probabilities = current_table_state / np.sum(current_table_state)
                action = np.random.choice(self.ACTIONS, p=probabilities)
                return action

        return None

    ################### storring and loading the q-table ########################
    def load_current_qtable(self,path_q_table,path_seen_rep):
        self.logger.info("loading Q-Table")
        self.Q_VALUES = np.load(path_q_table)
        self.SEEN_FEATURES = np.load(path_seen_rep)
        self.logger.info("loaded Q-Table")

    def create_qtable(self):
        self.Q_VALUES = np.ndarray(shape=(0,self.NUMBER_POSSIBLE_MOVES))
        self.SEEN_FEATURES = np.ndarray(shape=(0, self.NUMBER_FEATURES))

    def store_current_qtbale(self,path_q_table,path_seen_rep):
        if self.Q_VALUES is not None:
            self.logger.info("Saving current state of the q-table")
            np.save(path_q_table, self.Q_VALUES)
            np.save(path_seen_rep, self.SEEN_FEATURES)

    def prevent_getting_stuck(self, state):
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
