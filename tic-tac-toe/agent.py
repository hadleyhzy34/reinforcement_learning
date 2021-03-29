import numpy as np

class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps #prob of choosing random action instead of greedy
        self.alpha = alpha #learning rate
        self.verbose = False
        self.state_history = []

    def setV(self,V):
        self.V = V

    def set_symbol(self,sym):
        self.sym = sym
    
    def set_verbose(self, v):
        #if true, will print values for each position on the board
        self.verbose = v

    def reset_history(self):
        self.stte_history = []


    def take_action(self,env):
        #choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        best_state = None
        if r < self.eps:
            #take random action
            if self.verbose:
                print('taking a random action')


            possible_moves = []

