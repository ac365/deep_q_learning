from networks.DQN import DQN
from agent.Memory import Memory

import math
import random
import torch as T
import torch.optim as optim
class Agent:
    def __init__(self, obsDims:list, actDims:list, **hyperparams:dict):
        self._steps = 0

        self.device    = T.device("mps" if T.backends.mps.is_available()
                                   else "cpu")
        self.policyNet = DQN(obsDims,actDims).to(self.device)
        self.targetNet = DQN(obsDims,actDims).to(self.device)
        self.optimizer = optim.AdamW(self.policyNet.parameters(),
                                     lr = self.lr, amsgrad=True)

        if "MEM_CAPACITY" in hyperparams.keys():
            self.stateMem  = Memory(hyperparams["MEM_CAPACITY"],*obsDims)
            self.rewardMem = Memory(hyperparams["MEM_CAPACITY"],*obsDims)
            self.actionMem = Memory(hyperparams["MEM_CAPACITY"],*obsDims)
        else:
            self.stateMem  = Memory(100000,*obsDims)
            self.rewardMem = Memory(100000) 
            self.actionMem = Memory(100000,*actDims) 

        if "LR" in hyperparams.keys():
            self.lr = hyperparams["LR"]
        else:
            self.lr = 1e-4
        if "TAU" in hyperparams.keys():
            self.tau = hyperparams["TAU"]
        else:
            self.tau = 0.005
        if "GAMMA" in hyperparams.keys():
            self.gamma = hyperparams["GAMMA"]
        else:
            self.gamma = 0.99
        if "EPS_END" in hyperparams.keys():
            self.epsEnd = hyperparams["EPS_END"]
        else:
            self.epsEnd = 0.05
        if "EPS_START" in hyperparams.keys():
            self.epsStart = hyperparams["EPS_START"]
        else:
            self.epsStart = 0.9
        if "EPS_DECAY" in hyperparams.keys():
            self.epsDecay = hyperparams["EPS_DECAY"]
        else:
            self.epsDecay = 1000
        if "BATCH_SIZE" in hyperparams.keys():
            self.batchSize = hyperparams["BATCH_SIZE"]
        else:
            self.batchSize = 128

    def act(self, observation):
        self.epsilon = self.epsEnd + (self.epsStart - self.epsEnd)*\
            math.exp(-1. * self._steps / self.epsDecay)
        self.steps += 1

        if random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.device)
            action = self.policyNet.forward(state)
        else:
            pass
        return action