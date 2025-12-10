

import random, util
from feature_extractors import TrafficLightExtractor
from agents import ReinforcementAgent


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent.
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        else: 
            return max([self.getQValue(state, action) for action in legalActions])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        returnAction = None
        if not legalActions:
            returnAction = None
        else:
            maxQValue = self.computeValueFromQValues(state)
            # Use a small tolerance for float comparison
            bestActions = [action for action in legalActions if self.getQValue(state, action) >= maxQValue - 1e-10]
            returnAction = random.choice(bestActions)
        return returnAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions) if legalActions else None
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        newQValue = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
        self.qValues[(state, action)] = newQValue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class TrafficApproximateQAgent(QLearningAgent):
    """
    Approximate Q-Learning Agent for Traffic Lights.
    """
    def __init__(self, **args):
        self.featExtractor = TrafficLightExtractor()
        QLearningAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        qValue = 0
        for feature in features:
            qValue += self.weights[feature] * features[feature]
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        prevWeights = self.getWeights()
        for feature in features:
            self.weights[feature] = prevWeights[feature] + self.alpha * difference * features[feature]

