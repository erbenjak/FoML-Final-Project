import numpy as np
from collections import deque
"""
    This class manages everything required to perform regression learning. 
    
    The concrete implementations are managed by the inheriting models,
    this will hopefully allow for more structured code structuring over
    the different models. As all other models will be based on this class
    one needs to take special care of the correctness of all it's methods. 
    
    This class mainly defines some very broad general classes. Which then need to be fully
    implemented for the different models.
    
    It also provides the ability to turn/mirror the playing-field allowing for faster learning as well as 
    giving the the models an undefined feel for the symmetry of the game.
"""

class BaseModel:

    # the logger is an essential part used for debugging. To not be
    # forced to always parse the logger it is stored locally.
    logger = None
    rotationMatrix = None
    CENTER = None
    WIDTH = -1
    HEIGHT = -1
    ACTIONS = ["UP","DOWN","LEFT","RIGHT","WAIT","BOMB"]

    # we need to setup some-action history
    SHORT_MEMORY_LENGTH = 5
    # gets extended after choosing an action contains
    # defining a max-length means, that we are not required to pop the elements
    memory_short = deque([],maxlen=SHORT_MEMORY_LENGTH)
    # gets extended after learning and also stores the memory
    memory_long = deque()

    def __init__(self,logger, WIDTH=17, HEIGHT=17, CENTER=(8,8)):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CENTER = CENTER
        # logger is initialized
        if logger is None:
            raise ValueError('A regression model requires a logger')
        self.logger = logger
        # rotation-matrix is intialized it can be used to look up the new coordinates after the rotation
        self.rotationMatrix = self.initRotationMatrix(WIDTH,HEIGHT,CENTER)

    def clean_up(self):
        self.memory_short.clear()
        self.memory_long.clear()
        return

    def add_move_to_memory(self, state_as_feature, chosen_action):
        tuple_to_add = (state_as_feature, chosen_action)
        self.memory_short.append(tuple_to_add)

    ############################# Abstract Headers #######################################################

    def playGame(self, state):
        """
        Uses the learned progress to play the game.
        """
        raise NotImplementedError("A model must be able to use it's learned data to play the game. ")

    def calculateReward(self, events):
        """
        Calculate the reward a agent receives for a certain events.
        The return may be any float.
        """
        raise NotImplementedError("A model must provide a methode to calculate it's reward. ")

    def performLearningSingleState(self, singleStateOld, singleStateNew, singleAction, reward):
        """
        Perform the learning for an single instance.
        """
        raise NotImplementedError("A model must provide a methode to calculate it's reward. ")

    def performLearningLastState(self, lastState, lastAction, reward):
        """
        Perform the learning for the last game state.
        """
        raise NotImplementedError("A model must provide a way to learn from it's last game state. ")

    def calculateFeaturesFromState(self, state):
        """
        Takes a gamestate and extracts the features which are required to take good decisions.
        """
        raise NotImplementedError("A model must provide a methode to turn a gamestate into features. ")

    def compute_additional_rewards(self, events, new_state, old_state):
        """
        Takes a set of events produced by the game engine and adds some custom events to be able to
        add some additional self-defined 'custom' events
        """
        raise NotImplementedError("A model must be able to calculate some additional events to make custom "
                                  "rewards possible.")

    ############################## State Multiplication Code #############################################
    def performEndOfGameLearning(self, last_game_state, last_action, events):
        """
        This methode will call the learning version for a single instance after successfully
        multiplying the states.
        """
        gameStates = self.multiply_game_state(last_game_state)

        reward = self.calculateReward(events)

        self.logger.info("calculated_reward end of game = " + str(reward))

        actions = self.multiply_action(last_action)

        for i in range(0,len(gameStates)):
            self.performLearningLastState(gameStates[i],actions[i],reward)


    def performLearning(self, stateOld, stateNew, action, events):
        """
        This methode will call the learning version for a single instance after successfully
        multiplying the states.
        """
        oldStates = self.multiply_game_state(stateOld)
        newStates = self.multiply_game_state(stateNew)

        events = self.compute_additional_rewards(events, stateNew, stateOld)

        reward = self.calculateReward(events)

        self.logger.info("calculated_reward = " + str(reward))

        actions = self.multiply_action(action)

        for i in range(0,len(oldStates)):
            self.performLearningSingleState(oldStates[i],newStates[i],actions[i],reward)


    def multiply_game_state(self, stateToMultiply):
        """
        A given game state is mirrored. The original state as well as the mirrored state are then turned
        by 90, 180 and 270 degrees. Mirroring is uses the y-axis as its mirror-axis. The turns are performed
        clockwise. Please note that one can not simply use the same reward and action for all resulting
        states - either the actions need to be mirrored/rotated as well OR the rewards need to be calculated
        afterwards.
        """
        mirroredState = self.mirrorState(stateToMultiply)

        state90 = self.turnState90(stateToMultiply)
        mirroredState90 = self.turnState90(mirroredState)

        state180 = self.turnState90(state90)
        mirroredState180 = self.turnState90(mirroredState90)

        state270 = self.turnState90(state180)
        mirroredState270 = self.turnState90(mirroredState180)

        return [stateToMultiply, mirroredState, state90, mirroredState90,
                state180, mirroredState180, state270, mirroredState270]


    def mirrorState(self, stateToMultiply):
        """
        FIND ALL ENTRIES OF THE STATE DICTIONARY LISTED BELLOW AS WELL AS A DECISION IF THEY NEED TO BE CHANGED:
         1. round           -> NO
         2. step            -> NO
         3. field           -> YES (use np.fliplr => runtime O(1) --> very good)
         4. bombs           -> YES please note structure --> [(coordinates) , timer] => runtime O(n)
         5. explosion_map   -> YES (use np.fliplr => runtime O(1) --> very good)
         6. coins           -> YES please note structure --> [(coordinates)] => runtime O(n)
         7. self            -> YES please note structure (str,int,int,(coordinates)) => runtime O(1)
         8. others          -> YES please note structure [(str,int,int,(coordinates))] => runtime O(3)
         9. user_input      -> NO
            --> The overall runtime is luckily quite short
        """

        mirroredState = stateToMultiply.copy()

        ### 3.
        mirroredState['field'] = np.fliplr(mirroredState['field'])

        ### 4.
        mirrored_bombs = []
        for bomb in mirroredState['bombs']:
            mirrored_bombs.append((self.mirror_coordinates(bomb[0]),bomb[1]))
        mirroredState['bombs'] = mirrored_bombs

        ### 5.
        mirroredState['explosion_map'] = np.fliplr(mirroredState['explosion_map'])

        ### 6.
        mirrored_coins = []
        for coin in mirroredState['coins']:
            mirrored_coins.append(self.mirror_coordinates(coin))
        mirroredState['coins'] = mirrored_coins

        ### 7.
        mirrored_own_position = self.mirror_coordinates(mirroredState['self'][3])
        mirroredState['self'] = (mirroredState['self'][0], mirroredState['self'][1],
                                mirroredState['self'][2],mirrored_own_position)

        ### 8.
        mirrored_opponents = []
        for opponent in mirroredState['others']:
            mirrored_opponent_position = self.mirror_coordinates(opponent[3])
            mirrored_opponents.append((opponent[0], opponent[1], opponent[2], mirrored_opponent_position))
        mirroredState['others'] = mirrored_opponents

        return mirroredState

    def turnState90(self, stateToMultiply):
        """
        FIND ALL ENTRIES OF THE STATE DICTIONARY LISTED BELLOW AS WELL AS A DECISION IF THEY NEED TO BE CHANGED:
         1. round           -> NO
         2. step            -> NO
         3. field           -> YES (use np.fliplr => runtime O(1) --> very good)
         4. bombs           -> YES please note structure --> [(coordinates) , timer] => runtime O(n)
         5. explosion_map   -> YES (use np.fliplr => runtime O(1) --> very good)
         6. coins           -> YES please note structure --> [(coordinates)] => runtime O(n)
         7. self            -> YES please note structure (str,int,int,(coordinates)) => runtime O(1)
         8. others          -> YES please note structure [(str,int,int,(coordinates))] => runtime O(3)
         9. user_input      -> NO
            --> The overall runtime is luckily quite short
        """

        turnedState = stateToMultiply.copy()

        ### 3.
        turnedState['field'] = np.rot90(turnedState['field'], axes=(1, 0))

        ### 4.
        turned_bombs = []
        for bomb in turnedState['bombs']:
            turned_bombs.append((self.turn_coordinates(bomb[0]), bomb[1]))
        turnedState['bombs'] = turned_bombs

        ### 5.
        turnedState['explosion_map'] = np.rot90(turnedState['explosion_map'], axes=(1, 0))

        ### 6.
        turned_coins = []
        for coin in turnedState['coins']:
            #self.logger.info("coin has following coordinates:" +str(coin))
            turned_coins.append(self.turn_coordinates(coin))
        turnedState['coins'] = turned_coins

        ### 7.
        turned_own_position = self.turn_coordinates(turnedState['self'][3])
        turnedState['self'] = (turnedState['self'][0], turnedState['self'][1],
                                 turnedState['self'][2], turned_own_position)

        ### 8.
        turned_opponents = []
        for opponent in turnedState['others']:
            mirrored_opponent_position = self.turn_coordinates(opponent[3])
            turned_opponents.append((opponent[0], opponent[1], opponent[2], mirrored_opponent_position))
        turnedState['others'] = turned_opponents

        return turnedState

    def mirror_coordinates(self, coordinates):
        """
        uses the y-axis as the mirroring-axis
        """
        x, y = coordinates
        new_x = self.WIDTH - x - 1
        return new_x, y

    def turn_coordinates(self, coordinates):
        """
        rotates the coordinates by 90 degree in a clockwise direction
        """
        # to improve performance the rotation  coordinates are calculated only once and then stored
        x,y = coordinates
        return (self.rotationMatrix[x][y])

    def initRotationMatrix(self ,WIDTH, HEIGHT, CENTER):
        rotMat = np.empty((WIDTH,HEIGHT), dtype=object)

        for i in range(0, WIDTH):
            for j in range(0, HEIGHT):
                #self.logger.info("initial value: "+str(rotMat[i][j]))
                rotMat[i][j] = self.calculateRotatedCoordinates(i, j, CENTER)

        return rotMat

    def calculateRotatedCoordinates(self, x, y, CENTER):
        """
        Calculates the new coordinates after a rotation of 90 degrees around the given center.
        """
        x_store = x
        y_store = y
        x_prime = x - CENTER[0]
        y_prime = y - CENTER[1]

        x_prime_new = int(round(x_prime * np.cos(90 * np.pi / 180) - y_prime * np.sin(90 * np.pi / 180)))
        y_prime_new = int(round(y_prime * np.cos(90 * np.pi / 180) + x_prime * np.sin(90 * np.pi / 180)))

        x = x_prime_new + CENTER[0]
        y = y_prime_new + CENTER[1]

        if x < 0 or y < 0:
            self.logger.info("negative coordinates were calculated fot the following inpout: "+str(x_store)+" - "+str(y_store)+ " results: "+ str(x)+ "-" + str(y))
        if x > 16 or y > 16:
            self.logger.info("too big coordinates were calculated fot the following inpout: " + str(x_store) + " - " + str(y_store) + " results: " + str(x) + "-" + str(y))

        return x, y

    @staticmethod
    def multiply_action(action):

        ### [stateToMultiply, mirroredState, state90, mirroredState90,
        ###        state180, mirroredState180, state270, mirroredState270]

        if action == "UP":
            return ["UP", "UP", "RIGHT", "RIGHT", "DOWN", "DOWN", "LEFT", "LEFT"]

        if action == "DOWN":
            return ["DOWN", "DOWN", "LEFT", "LEFT", "UP", "UP", "RIGHT", "RIGHT"]

        if action == "LEFT":
            return ["LEFT", "RIGHT", "UP", "DOWN", "RIGHT", "LEFT", "DOWN", "UP"]

        if action == "RIGHT":
            return ["RIGHT", "LEFT", "DOWN", "UP", "LEFT", "RIGHT", "UP", "DOWN"]

        if action == "WAIT":
            return ["WAIT", "WAIT", "WAIT", "WAIT", "WAIT", "WAIT", "WAIT", "WAIT"]

        if action == "BOMB":
            return ["BOMB", "BOMB", "BOMB", "BOMB", "BOMB", "BOMB", "BOMB", "BOMB"]

    ################################## Action-Handeling #########################

    def action_to_index(self, action):
        return self.ACTIONS.index(action)

    def index_to_action(self, actionIndex):
        return self.ACTIONS[actionIndex]

    def getActions(self):
        return self.ACTIONS