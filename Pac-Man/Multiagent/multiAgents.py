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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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

        "*** YOUR CODE HERE ***"
        position =newFood.asList()
        count =len(position)
        close_distance =1e6
        for i in range(count):
          distance=manhattanDistance(position[i],newPos)+count*100
          if distance<close_distance:
            close_distance =distance
            food=position
        if count ==0 :
          close_distance =0
        score = -close_distance

        for i in range(len(newGhostStates)):
          ghostPos =successorGameState.getGhostPosition(i+1)
          if manhattanDistance(newPos,ghostPos)<=1 :
            score -=1e6

        return score 

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
        "*** YOUR CODE HERE ***"
        agent_cnt = gameState.getNumAgents()
        ActionScore = []

        def stop(List):
          return [x for x in List if x != 'Stop']

        def min_max(state, count):
          if count >= self.depth * agent_cnt or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if count % agent_cnt != 0: 
            result = 1e10
            for i in stop(state.getLegalActions(count % agent_cnt)):
              curr = state.generateSuccessor(count % agent_cnt,i)
              result = min(result, min_max(curr, count+1))
            return result
          else: 
            result = -1e10
            for i in stop(state.getLegalActions(count % agent_cnt)):
              curr = state.generateSuccessor(count % agent_cnt,i)
              result = max(result, min_max(curr, count + 1))
              if count == 0:
                ActionScore.append(result)
            return result
          
        result = min_max(gameState, 0);
        return stop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agents = gameState.getNumAgents()
        ActionScore = []

        def stop(List):
          return [x for x in List if x != 'Stop']

        def alpha_Beta(state, count, alpha, beta):
          if count >= self.depth * agents or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if count % agents != 0: 
            result = 1e10
            for a in stop(state.getLegalActions(count % agents)):
              curr = state.generateSuccessor(count % agents,a)
              result = min(result, alpha_Beta(curr, count+1, alpha, beta))
              beta = min(beta, result)
              if beta < alpha:
                break
            return result
          else: 
            result = -1e10
            for a in stop(state.getLegalActions(count % agents)):
              curr = state.generateSuccessor(count % agents,a)
              result = max(result, alpha_Beta(curr, count + 1, alpha, beta))
              alpha = max(alpha, result)
              if count == 0:
                ActionScore.append(result)
              if beta < alpha:
                break
            return result

        result = alpha_Beta(gameState, 0, -1e20, 1e20)
        return stop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]

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
        "*** YOUR CODE HERE ***"
        agents = gameState.getNumAgents()
        ActionScore = []

        def stop(List):
          return [x for x in List if x != 'Stop']

        def expect_Minimax(state, count):
          if count >= self.depth * agents or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if count % agents != 0: 
            successorScore = []
            for a in stop(state.getLegalActions(count % agents)):
              curr = state.generateSuccessor(count % agents,a)
              result = expect_Minimax(curr, count + 1)
              successorScore.append(result)
            averageScore = sum([ float(x)/len(successorScore) for x in successorScore])
            return averageScore
          else:
            result = -1e10
            for a in stop(state.getLegalActions(count % agents)):
              curr = state.generateSuccessor(count%agents,a)
              result = max(result, expect_Minimax(curr, count+1))
              if count == 0:
                ActionScore.append(result)
            return result
          
        result = expect_Minimax(gameState, 0);
        return stop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def score_winfrom_food(gameState):
      dist_from_food = []
      for food in gameState.getFood().asList():
        dist_from_food.append(1.0/manhattanDistance(gameState.getPacmanPosition(), food))
      if len(dist_from_food)>0:
        return max(dist_from_food)
      else:
        return 0  

    def score_winfrom_ghost(gameState):
      score = 0
      for ghost in gameState.getGhostStates():
        dist_from_ghost = manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition())
        if ghost.scaredTimer > 0:
          score += pow(max(8 - dist_from_ghost, 0), 2)
        else:
          score -= pow(max(7 - dist_from_ghost, 0), 2)
      return score
    
    def score_winfrom_cap(gameState):
      score = []
      for cap in gameState.getCapsules():
        score.append(50.0/manhattanDistance(gameState.getPacmanPosition(), cap))
      if len(score) > 0:
        return max(score)
      else:
        return 0    

    def self_termination(gameState):
      score = 0
      dist_from_ghost = 1e6
      for ghost in gameState.getGhostStates():
        dist_from_ghost = min(manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition()), dist_from_ghost)
      score -= pow(dist_from_ghost, 2)
      if gameState.isLose():
        score = 1e6
      return score    

    score = currentGameState.getScore()
    ghost_score = score_winfrom_ghost(currentGameState)
    food_score = score_winfrom_food(currentGameState)
    cap_score = score_winfrom_cap(currentGameState)
    
    return score + ghost_score + food_score + cap_score    

# Abbreviation
better = betterEvaluationFunction
