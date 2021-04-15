import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        #the first capacity-1 positions are not leaves
        self.vals = [0 for _ in range(2*capacity - 1)]

    def retrive(self, num):
        '''
        find the first index who sum is not smaller than num
        '''
        ind = 0
        
        return ind - self.capacity + 1

    def update(self, delta, ind):
        '''
        change the value at ind by delta, and update the tree
        notice that this ind should be the index in real memory part, instead of ind in self.vals
        '''
        ind += self.capacity - 1
        while True:
            self.vals[ind] += delta
            if ind == 0:
                break
            ind -= 1
            ind //= 2
