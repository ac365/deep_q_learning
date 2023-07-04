import random
import numpy as np


class Memory:
    def __init__(self, capacity, *inpDims) -> None:
        self.index    = 0
        self.capacity = capacity
        self.memory   = np.zeros([capacity, *inpDims], dtype=np.float32)
    
    def __len__(self): #TODO: delete if unused
        return len(self.memory)
    
    def push(self, data) -> None:
        self.index              = self.index % self.capacity
        self.memory[self.index] = data
        self.index             += 1
   
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    