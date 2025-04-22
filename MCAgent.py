import numpy as np

# class for the Monte Carlo agent for the Blackjack game
class MCAgent:
    def __init__(self, eps = 0.3, gamma = 0.98):
        self.eps = eps
        self.gamma = gamma
        self.Q = {} # action-value function for (s, a) pairs
        self.returns = {} # returns for (s, a) pairs
        self.policy = {}

        self.action_space = [0, 1] # either stick or hit
        self.sum_space = list(range(4, 22)) # possible values for the sum of the player's hand
        self.dealer_space = list(range(1, 11))
        self.ace_space = [0, 1]

        self.state_space = [] # all possible states (sum, dealer, ace)
        self.init_state_action_values()
    
    def init_state_action_values(self):
        # initialize the state-action values and returns for all possible states
        for total in self.sum_space:
            for dealer_card in self.dealer_space:
                for usable_ace in self.ace_space:
                    state = (total, dealer_card, usable_ace)
                    self.state_space.append(state)

                    for action in self.action_space:
                        self.Q[(state, action)] = 0.0
                        self.returns[(state, action)] = []

                    # uniform random policy ie hit = 0.5 stick = 0.5
                    self.policy[state] = [0.5, 0.5] # basically a list of probabilities for each action
    
    def choose_action(self, state):
        # choose an action based on the epsilon-greedy policy
        if state not in self.policy:
            # if state not in policy, act randomly
            return np.random.choice(self.action_space)
        
        action_probs = self.policy[state]
        return np.random.choice(self.action_space, p=action_probs)
        

    def update_q(self):
        G = 0 # return to go func
        visited = set() # to keep track of visited states in the episode

        for t in reversed(range(len(self.episode))): # going from final step to first step
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                self.returns[(state, action)].append(G)
                self.Q[(state, action)] = np.mean(self.returns[(state, action)]) # q function takes expected val
                self.update_policy(state)
                

    def update_policy(self, state):
        q_vals = [self.Q[(state, action)] for action in self.action_space]
        best_q = np.argmax(q_vals) # index of the best action
        self.policy[state] = [self.eps/2, self.eps/2] # initialize policy with epsilon
        self.policy[state][best_q] += 1 - self.eps # so that prob sum is 1

