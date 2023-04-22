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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            iterationValues = util.Counter()
            iterationStates = self.mdp.getStates()
            for state in iterationStates:
                if self.mdp.isTerminal(state):
                    self.values[state] = self.mdp.getReward(state, "exit", "")
                else:
                    actions = self.mdp.getPossibleActions(state)
                    iterationValues[state] = max([self.computeQValueFromValues(state, action) for action in actions])
            self.values = iterationValues
        


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

        actions = self.mdp.getTransitionStatesAndProbs(state, action)
        totalValue = 0

        for nextState, probability in actions:
            reward = self.mdp.getReward(state, action, nextState)
            totalValue += probability*(reward + self.discount*self.values[nextState]) # end of the bellman equation
        return totalValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Iterate through each value, action combo
        actionValue = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            actionValue[action] = self.computeQValueFromValues(state, action)
        return actionValue.argMax()


        
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        iterationStates = self.mdp.getStates()

        for i in range(self.iterations):
            state = iterationStates[i%len(iterationStates)]
            if self.mdp.isTerminal(state):
                self.values[state] = self.mdp.getReward(state, "exit", "")
            else:
                action = self.getAction(state)
                value = self.getQValue(state, action)
                self.values[state] = value

        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Compute predecessors of all states.
        allStates = self.mdp.getStates()
        # for state in allStates:
        #     predStates = list()
        #     for secondState in allStates:
        #         if secondState!=state:
        #             actions = self.mdp.getPossibleActions(secondState)
        #             for action in actions:
        #                 transitionActions = self.mdp.getTransitionStatesAndProbs(self, secondState, action)
        #                 for nextState, prob in transitionActions:
        #                     if nextState == state and prob > 0:
        #                         predStates.add(secondState, action)
            
            
        
        # Initialize an empty priority queue.
        pQueue = util.PriorityQueue()

        #For each non-terminal state s, do:
        for state in allStates:
            if self.mdp.isTerminal(state):
                # self.values[state] = self.mdp.getReward(state, "exit", "")
                continue
            else:
                actions = self.mdp.getPossibleActions(state)
                actionValue = max([self.getQValue(state, action) for action in actions])
                diff = abs(self.values[state] - actionValue)
                # Push s into the priority queue with priority -diff 
                pQueue.push(state, -diff)
        
        for i in range(self.iterations):
            if pQueue.isEmpty():
                return
            curState = pQueue.pop()
            # if self.mdp.isTerminal(curState):
            #     continue
            # else:
            curStateActions = self.mdp.getPossibleActions(curState)
            maxCurValue = max([self.getQValue(curState, action) for action in curStateActions])
            self.values[curState] = maxCurValue
            curStatePreds = self.getPredecessors(curState)
            for predState in curStatePreds:
                predStateActions = self.mdp.getPossibleActions(predState)
                predStateValue = max([self.getQValue(predState, action) for action in predStateActions])
                diff = abs(self.values[predState] - predStateValue)
                if diff > self.theta:
                    pQueue.update(predState, -diff)

        # For each predecessor p of s, do:



    def getPredecessors(self, state):
        predStates = set() #list or set?
        allStates = self.mdp.getStates()
        if not self.mdp.isTerminal(state):
            for secondState in allStates:
                if not self.mdp.isTerminal(secondState):
                    actions = self.mdp.getPossibleActions(secondState)
                    for action in actions:
                        transitionActions = self.mdp.getTransitionStatesAndProbs(secondState, action)
                        for nextState, prob in transitionActions:
                            if nextState == state and prob > 0:
                                predStates.add(secondState)
        return predStates
        