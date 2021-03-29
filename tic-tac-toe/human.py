from builtins import range,input
import numpy as np


class Human:
    def __init__(self):
        pass

    def set_symbol(self,sym):
        self.sym = sym

    def take_action(self, env):
        while True:
            #break if we make a legal move
            move = input("Enter coordinated i,j for your next move: ")
            i,j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i,j):
                env.board[i,j] = self.sym
                break
            else:
                print('your move is not correct')

    def update(self,env):
        pass

    def update_state_history(self, s):
        pass




