    # valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        while self.iterations > 0:
            self.iterations = self.iterations - 1
            possibleSt = mdp.getStates()
            copiedValues = self.values.copy()
            for eachSt in possibleSt:
                chances = []
                possibleAt = mdp.getPossibleActions(eachSt)
                for eachAt in possibleAt:
                    avg = 0
                    currSt = mdp.getTransitionStatesAndProbs(eachSt,eachAt)
                    for eachCurrSt in currSt:
                        nextSt = eachCurrSt[0]
                        alpha = eachCurrSt[1]
                        reward = mdp.getReward(eachSt,eachAt,nextSt)
                        gamma = discount * copiedValues[nextSt]
                        nextStateVal = alpha * (reward + gamma)
                        avg = avg + nextStateVal
                    chances.append(avg)
                length = len(chances)
                if length > 0:
                    max_chances = max(chances)
                    self.values[eachSt] = max_chances
                
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        avg = 0
        stAndPr = self.mdp.getTransitionStatesAndProbs(state,action)
        for eachSt in stAndPr:
            currSt = eachSt[0]
            alpha = eachSt[1]
            reward = self.mdp.getReward(state,action,currSt)
            gamma = self.values[currSt] * self.discount
            nextStateVal = alpha * (gamma + reward)
            avg = avg + nextStateVal
        return avg

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        takenAt = ""
        sum = float("-inf")
        allAt = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state):
            return None
        else:
            for eachAt in allAt:
                avg = self.computeQValueFromValues(state,eachAt)
                if sum == 0.0:
                    if eachAt == "":
                        takenAt = eachAt
                        sum = avg
                elif sum <= avg:
                    takenAt = eachAt
                    sum = avg
        return takenAt


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
