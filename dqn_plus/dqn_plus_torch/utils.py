import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity # number of leaf nodes
        #the first capacity-1 positions are not leaves

        # generate the trees with all nodes values = 0
        # parent_nodes = capacity -1, leaf_nodes = capacity
        self.vals = [0 for _ in range(2*capacity - 1)]

    def retrive(self, num):
        '''
        find the !first index who sum is not smaller than num
        '''
        ind = 0 #seraching from root
        #print(f'current num is: {num}')
        while ind < self.capacity -1: # searching non-leaf node
            left = 2*ind + 1
            right = left + 1
            if num > self.vals[left]:
                num -= self.vals[left]
                ind = right
            else: #search in the left tree
                ind = left
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
