import numpy as np


# this class represents a tic-tac-toe game
class Environment:
    def __init__(self,length):
        self.length = length
        self.board = np.zeros((self.length,self.length))
        self.x = -1
        self.o = 1
        self.winner = None
        self.ended = False
        self.num_states = 3**(self.length*self.length)

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
        for i in range(self.length):
            for j in range(self.length):
                if self.board[i,j] == 0:
                    v = 0
                elif self.board[i,j] == self.x:
                    v = 1
                elif self.board[i,j] == self.o:
                    v = 2
                h += (3**k) * v
                k +=1
        return h

    def get_state_1(self):
        #returns the current state, represented as an int
        state = 0
        for i in range(self.length):
            for j in range(self.length):
                if self.board[i,j] == 0:
                    state += 0*(i*self.length + j)*10
                if self.board[i,j] == 1:
                    state += 1*(i*self.length + j)*10
                if self.board[i,j] == 2:
                    state += 2*(i*self.length + j)*10
        return state

    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended

        #check rows
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = true
                    return True

        #check columns
        for j in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[:,j].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        # check diagonals
        for player in (self.x, self.o):
        # top-left -> bottom-right diagonal
            if self.board.trace() == player*LENGTH:
                self.winner = player
                self.ended = True
                return True
        # top-right -> bottom-left diagonal
            if np.fliplr(self.board).trace() == player*LENGTH:
                self.winner = player
                self.ended = True
                return True

        # check if draw
            if np.all((self.board == 0) == False):
            # winner stays None
                self.winner = None
                self.ended = True
                return True

        # game is not over
            self.winner = None
            return False

    def is_draw(self):
        return self.ended and self.winner is None


    def draw_board(self):
        for i in range(self.length):
            print("-------------")
            for j in range(self.length):
                print("  ", end="")
                if self.board[i,j] == self.x:
                    print("x ", end="")
                elif self.board[i,j] == self.o:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print("-------------")





if __name__ == '__main__':
    test = Environment(3)
    test.draw_board()
    print(f'current state of board is: {test.get_state()}')
