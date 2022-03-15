import events as e
import numpy as np
import os.path
from .BaseModel import BaseModel
"""
    Manages the Basic Q-Learning processes while leaving the feature and reward calculations to the concreate agent
    implementations.  
"""
class BaseQLearningModel(BaseModel):

    Q_TABLE = None
    ALPHA = 0.15
    GAMMA = 0.10
    EPSILON = 0.05
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

    def get_epsilon(self):
        return self.EPSILON

    ################ playing the actual game with the given information ###########
    def playGame(self,state):
        """plays the game using the trained qtable"""
        feature_state = self.calculateFeaturesFromState(state)
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
        return self.ACTIONS[int(move)]

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