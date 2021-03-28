import numpy as np


# this class represents a tic-tac-toe game
class Environment:
    def __init__(self):
        self.board = np.zeros(LENGTH,LENGTH)
        self.x = -1
        self.o = 1
        self.winner = None
        self.ended = False
        self.num_states = 3**(LENGTH*LENGTH)

    def is_empty(self, i ,j):
        return self.board[i,j] == 0
    
    def reward(self, sym):
        #no reward until game is over
        if not self.game_over():
            return 0

        #if game is over
        return 1 if self.winner == sym else 0

    def get_state(self):
        #returns the current state, represented as an int
        k = 0
        h = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i,j] == 0:
                    v = 0
                elif

