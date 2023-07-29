import random
import numpy as np
from typing import Type


class Memory:
    def __init__(self, capacity:int, dtype:Type, *inpDims:tuple) -> None:
        self.index    = 0
        self.capacity = capacity
        self.memory   = np.zeros([capacity, *inpDims], dtype)
    
    def __len__(self) -> int: #TODO: delete if unused
        return len(self.memory)
    
    def push(self, data) -> None: #TODO: can I not specify datatype?
        self.index              = self.index % self.capacity
        self.memory[self.index] = data
        self.index             += 1
   
    def sample(self, batch_size:int):
        indices = np.random.choice(self.index, batch_size, replace=False)
        return self.memory[indices]
    