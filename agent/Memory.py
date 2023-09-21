import numpy as np
from typing import Type


class Memory:
    def __init__(self, capacity:int, dtype:Type, *inpDims:tuple) -> None:
        self.counter  = 0
        self.capacity = capacity
        self.memory   = np.zeros([capacity, *inpDims], dtype)
    
    def __len__(self) -> int: #TODO: delete if unused
        return len(self.memory)
    
    def push(self, data) -> None: #TODO: can I not specify datatype?
        index              = self.counter % self.capacity
        self.memory[index] = data
        self.counter      += 1
   
    def sample(self, batchSize:int, seed:int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        memSize = min(self.counter, len(self.memory))
        indices = rng.choice(memSize, batchSize, replace=False)
        return self.memory[indices]
    