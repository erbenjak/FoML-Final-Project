import events as e
import numpy as np
import os.path
from .BaseModel import BaseModel
import random
import math
"""
    Manages the Basic Q-Learning processes while leaving the feature and reward calculations to the concreate agent
    implementations.  
"""
class BaseQLearningModel(BaseModel):

    Q_TABLE = None
    ALPHA = 0.15
    GAMMA = 0.10
    EPSILON_TRESHHOLD = 0.05
    EPSILON_START = 1.1
    EPSILON_DECAY_PARAM_1 = 0.5
    EPSILON_DECAY_PARAM_2 = 0.2
    EPSILON_DECAY_PARAM_3 = 0.1
    NUM_ROUNDS_TRAINING = 100

    #the max_feature_size determines all possible featurized states
    max_feature_size = -1
    path = None

    def __init__(self, logger, max_feature_size,path, ALPHA = 0.15, GAMMA = 0.10, EPSILON = 0.05):
        BaseModel.__init__(self,logger)
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.max_feature_size=max_feature_size
        self.path = path

    ################## setup ####################################################
    def setup(self):
        self.logger.info("Setting up the q-learning agent")
        if os.path.isfile(self.path) == False:
            self.create_qtable()
        else:
            self.load_current_qtable(self.path)

    ################# progress ##################################################
    def store_progress(self):
        self.logger.info("Storing learning progress")
        self.store_current_qtbale(self.path)

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
        index_old = self.state_to_index(features_old)
        index_new = self.state_to_index(features_new)
        action_index = self.action_to_index(singleAction)
        # performe the q-learning process
        followup_reward = np.amax(self.Q_TABLE[index_new])
        self.Q_TABLE[index_old][action_index] = (1 - self.ALPHA) * self.Q_TABLE[index_old][action_index] + \
                                                self.ALPHA * (reward + self.GAMMA * followup_reward)

    def performLearningLastState(self, lastState, lastAction, reward):
        """
        Perform the learning for the last game state.
        """
        followup_reward=0

        features = self.calculateFeaturesFromState(lastState)
        index = self.state_to_index(features)
        action_index = self.action_to_index(lastAction)

        self.Q_TABLE[index][action_index] = (1 - self.ALPHA) * self.Q_TABLE[index][action_index] + \
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
        feature_state = self.calculateFeaturesFromState(state)

        if train:
            if random.random() < self.get_epsilon(state['round']):
                action_chosen = self.getActions()[int(random.randint(0, 5))]
                self.add_move_to_memory(feature_state, action_chosen)
                return action_chosen

            ###prevent getting stuck:
            action = self.preventOverLearning(state)
            if action is not None:
                return action

        """plays the game using the trained qtable"""
        index = self.state_to_index(feature_state)
        move_values = self.Q_TABLE[int(index)]

        #self.logger.info("decision function knows the following move values" + str(move_values))
        move = np.argmax(move_values)

        # in case the highest value appears more than once choose a random action
        move_val = move_values[move]
        if np.where(move_values == move_val)[0].shape[0] > 1:
            possible_move_indecies = np.where(move_values == move_val)[0]
            #self.logger.info("Choosing from the follwoing values: " + str(possible_move_indecies))
            np.random.shuffle(possible_move_indecies)
            move = possible_move_indecies[0]
        chosen_action = self.ACTIONS[int(move)]
        self.add_move_to_memory(feature_state, chosen_action)

        #self.logger.info("The current short memory looks as follows:")
        #self.logger.info(self.memory_short)
        return chosen_action

    def preventOverLearning(self, state):
        """
        We want to adrestraining. In these cases we are aware that the same
        will pop up continuously only to be broken up by as to special cases in order to improve our n exploration. In order to prevent our model from
        overlearning certain states. We allow the agent to then take a probabistic approach to the next step. Allowing
        us to live carry on.
        """
        feature = self.calculateFeaturesFromState(state)
        feature_ind = self.state_to_index(feature)

        short_term_memory = self.memory_short.copy()
        if len(short_term_memory) > 4:
            hist1 = short_term_memory.pop()
            hist2 = short_term_memory.pop()
            hist3 = short_term_memory.pop()
            hist4 = short_term_memory.pop()

            back_and_forth = (hist1[1] == hist3[1] and hist2[1] == hist4[1] and hist1[1] != hist2[1])
            waiting = (np.all(hist1[0] == hist2[0]) and hist1[1] == hist2[1] and hist1[1]=='WAIT')
            if back_and_forth or waiting:
                current_table_state = self.Q_TABLE[feature_ind]
                #self.logger.info(current_table_state)
                ### if the current state only contains 0s then we would divide by 0
                if(np.all(current_table_state==0)):
                    current_table_state = current_table_state + 1
                ### ckeck if negative values exist is so add smallest value to all values twice
                #self.logger.info(2 * np.min(current_table_state))
                if np.any(current_table_state < 0):
                    current_table_state = current_table_state + (2 * abs(np.min(current_table_state)))
                ### preventing the dropping of random bombs
                #self.logger.info(current_table_state)
                current_table_state[5] = 0
                ### probabilities are determined by dividing through the sum
                probabilities = current_table_state / np.sum(current_table_state)
                #self.logger.info("Calculated Probabilities : "+str(probabilities))
                action = np.random.choice(self.ACTIONS, p=probabilities)
                #self.logger.info("Prevented a learning loop via a softmax function")
                return action

        return None

    ################### storring and loading the q-table ########################
    def load_current_qtable(self,abs_path):
        self.logger.info("loading Q-Table")
        self.Q_TABLE = np.load(abs_path)
        self.logger.info("loaded Q-Table")

    def create_qtable(self):
        self.Q_TABLE = np.empty((self.max_feature_size, len(self.ACTIONS)))

    def store_current_qtbale(self):
        if self.Q_TABLE is not None:
            self.logger.info("Saving current state of the q-table")
            np.save(self.path, self.Q_TABLE)