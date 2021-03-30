import numpy as np
from environment import Environment


class Agent:
    def __init__(self, eps=0.1, alpha=0.5, length=3):
        self.eps = eps #prob of choosing random action instead of greedy
        self.alpha = alpha #learning rate
        self.verbose = False
        self.length = 3
        self.state_history = []

    def setV(self,V):
        #initialize states
        self.V = V

    def set_symbol(self,sym):
        self.sym = sym
    
    def set_verbose(self, v):
        #if true, will print values for each position on the board
        self.verbose = v

    def reset_history(self):
        self.state_history = []


    def take_action(self,env):
        #choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        best_state = None
        if r < self.eps:
            #take random action
            if self.verbose:
                print('taking a random action')


            possible_moves = []
            for i in range(self.length):
                for j in range(self.length):
                    if env.is_empty((i,j)):
                        possible_moves.append((i,j))
            #choose one of possible moves using uniform possibility
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
        # choose the best action based on current values of states
        # loop through all possible moves, get their values
        #  keep track of the best value
            next_move = None
            best_value = -1
            for i in range(self.length):
                for j in range(self.length):
                    if env.is_empty(i,j):
                        #what is the state if we made this move?
                        env.board[i,j] = self.sym
                        state = env.get_state()
                        env.board[i,j] = 0 #don't forget to change it back
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i,j)
        env.board[next_move[0], next_move[1]] = self.sym


    def update_state_history(self,s):
        self.state_history.append(s)

    def update(self,env):
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()


if __name__ == '__main__':
    test = Agent()
    test1 = Environment(3)
    test.set_symbol(test1.x)
    test.take_action(test1)
