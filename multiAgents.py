# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print "action: " + str(legalMoves[chosenIndex])

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        min_food_dist = 9999999
        newFood = newFood.asList()
        for food in newFood:
            if util.manhattanDistance(newPos, food) < min_food_dist:
                min_food_dist = util.manhattanDistance(newPos, food)
        # print "min_food_dist: " + str(min_food_dist)
        max_food_dist = 0
        for food in newFood:
            if util.manhattanDistance(newPos, food) > max_food_dist:
                max_food_dist = util.manhattanDistance(newPos, food)
        closest_ghost_dist = 9999999
        for ghost in newGhostStates:
            if util.manhattanDistance(newPos, ghost.getPosition()) < closest_ghost_dist:
                closest_ghost_dist = util.manhattanDistance(newPos, ghost.getPosition())
        # print "closest_ghost_dist: " + str(closest_ghost_dist)
        evaluationValue = (1.0 / (min_food_dist + 1.0)) + (successorGameState.getScore()) - (1.0 / (closest_ghost_dist + 1.0))
        if closest_ghost_dist == 0 or closest_ghost_dist == 1:
            evaluationValue = -99999
        # print "evaluationValue: " + str(action) + str(evaluationValue)
        return evaluationValue

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        num_of_agents = gameState.getNumAgents()
        def minimaxHelper(gameState, depth, agent):
            def value(gameState, depth, agent):
                if gameState.isWin() or gameState.isLose() or depth == 0:
                    return self.evaluationFunction(gameState), None
                if agent == 0:
                    return maxValue(gameState, depth, agent)
                return minValue(gameState, depth, agent)

            def maxValue(gameState, depth, agent):
                v = -99999999
                a = None
                for action in gameState.getLegalActions(agent):
                    val = value(gameState.generateSuccessor(agent, action), depth, (agent + 1) % num_of_agents)[0]
                    if val > v:
                        v = val
                        a = action
                return v, a

            def minValue(gameState, depth, agent):
                v = 99999999
                a = None
                for action in gameState.getLegalActions(agent):
                    if (agent + 1) % num_of_agents == 0:
                        val = value(gameState.generateSuccessor(agent, action), depth - 1, (agent + 1) % num_of_agents)[0]
                    else:
                        val = value(gameState.generateSuccessor(agent, action), depth, (agent + 1) % num_of_agents)[0]
                    if val < v:
                        v = val
                        a = action
                return v, a
            return value(gameState, depth, 0)
        return minimaxHelper(gameState, self.depth, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        num_of_agents = gameState.getNumAgents()
        def minimaxHelper(gameState, depth, agent, a, b):
            def value(gameState, depth, agent, a, b):
                if gameState.isWin() or gameState.isLose() or depth == 0:
                    return self.evaluationFunction(gameState), None
                if agent == 0:
                    return maxValue(gameState, depth, agent, a, b)
                return minValue(gameState, depth, agent, a, b)

            def maxValue(gameState, depth, agent, a, b):
                v = -99999999
                act = None
                for action in gameState.getLegalActions(agent):
                    val = value(gameState.generateSuccessor(agent, action), depth, (agent + 1) % num_of_agents, a, b)[0]
                    if val > b:
                        return val, action
                    if val > v:
                        v = val
                        act = action
                    a = max(a, v)
                return v, act

            def minValue(gameState, depth, agent, a, b):
                v = 99999999
                act = None
                for action in gameState.getLegalActions(agent):
                    if (agent + 1) % num_of_agents == 0:
                        val = value(gameState.generateSuccessor(agent, action), depth - 1, (agent + 1) % num_of_agents, a, b)[0]
                    else:
                        val = value(gameState.generateSuccessor(agent, action), depth, (agent + 1) % num_of_agents, a, b)[0]
                    if val < a:
                        return val, action
                    if val < v:
                        v = val
                        act = action
                    b = min(b, v)
                return v, act
            return value(gameState, depth, 0, a, b)
        return minimaxHelper(gameState, self.depth, 0, -99999999, 99999999)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        num_of_agents = gameState.getNumAgents()
        def minimaxHelper(gameState, depth, agent):
            def value(gameState, depth, agent):
                if gameState.isWin() or gameState.isLose() or depth == 0:
                    return self.evaluationFunction(gameState), None
                if agent == 0:
                    return maxValue(gameState, depth, agent)
                return avgValue(gameState, depth, agent), None

            def maxValue(gameState, depth, agent):
                v = -99999999
                a = None
                for action in gameState.getLegalActions(agent):
                    val = value(gameState.generateSuccessor(agent, action), depth, (agent + 1) % num_of_agents)[0]
                    if val > v:
                        v = val
                        a = action
                return v, a

            def avgValue(gameState, depth, agent):
                v = 0
                for action in gameState.getLegalActions(agent):
                    if (agent + 1) % num_of_agents == 0:
                        val = value(gameState.generateSuccessor(agent, action), depth - 1, (agent + 1) % num_of_agents)[0]
                    else:
                        val = value(gameState.generateSuccessor(agent, action), depth, (agent + 1) % num_of_agents)[0]
                    v += val
                return float(v) / len(gameState.getLegalActions(agent))
            return value(gameState, depth, 0)

        return minimaxHelper(gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    min_food_dist = 9999999
    food = food.asList()
    for f in food:
        if util.manhattanDistance(pos, f) < min_food_dist:
            min_food_dist = util.manhattanDistance(pos, f)
    min_cap_dist = 9999999
    for c in capsules:
        if util.manhattanDistance(pos, c) < min_cap_dist:
            min_cap_dist = util.manhattanDistance(pos, c)
    scared_time_totals = reduce(lambda x, y: x + y, scaredTimes, 0)
    #print "min_food_dist: " + str(min_food_dist)
    max_food_dist = 0
    for f in food:
        if util.manhattanDistance(pos, f) > max_food_dist:
            max_food_dist = util.manhattanDistance(pos, f)
    closest_ghost_dist = 9999999
    for ghost in ghostStates:
        if util.manhattanDistance(pos, ghost.getPosition()) < closest_ghost_dist:
            closest_ghost_dist = util.manhattanDistance(pos, ghost.getPosition())
    #print "closest_ghost_dist: " + str(closest_ghost_dist)
    
    evaluationValue = (1.0 / (min_food_dist + 1.0)) + (currentGameState.getScore()) - (1.0 / (closest_ghost_dist + 1.0)) + (1.0 / (min_cap_dist + 1.0)) + scared_time_totals
    if closest_ghost_dist == 0 or closest_ghost_dist == 1:
        evaluationValue = -99999
    #print "evaluationValue: " + str(evaluationValue)
    return evaluationValue

# Abbreviation
better = betterEvaluationFunction

