from networks.DQN import DQN as dqn
from agent.Memory import Memory as mem
import gymnasium as gym #FIXME? is this really the best way to do this?

import math
import random
import numpy as np
import torch as T
import torch.optim as optim

class Agent:
    def __init__(self, env:gym.Env, obsDims:list, actDims:list,
                  **hyperparams:dict):
        #member variables
        self._env    = env
        self._steps  = 0
        self._device = T.device("mps" if T.backends.mps.is_available()
                                   else "cpu")

        #kwargs
        if "MEM_CAPACITY" in hyperparams.keys():
            self.stateMem     = mem(hyperparams["MEM_CAPACITY"],*obsDims,
                                    np.float32)
            self.nextStateMem = mem(hyperparams["MEM_CAPACITY"],*obsDims,
                                    np.float32)
            self.actionMem    = mem(hyperparams["MEM_CAPACITY"],*actDims,
                                    np.int32)
            self.rewardMem    = mem(hyperparams["MEM_CAPACITY"],
                                    np.float32)
            self.doneMem      = mem(hyperparams["MEM_CAPACITY"],
                                    np.bool_)
        else:
            self.stateMem     = mem(100000,*obsDims,*obsDims,np.float32)
            self.nextStateMem = mem(100000,*obsDims,*obsDims,np.float32)
            self.actionMem    = mem(100000,*actDims,*actDims,np.int32) 
            self.rewardMem    = mem(100000,dtype=np.float32) 
            self.doneMem      = mem(100000,dtype=np.bool_)

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

        #networks
        self.policyNet = dqn(obsDims,actDims).to(self._device)
        self.targetNet = dqn(obsDims,actDims).to(self._device)
        self.optimizer = optim.AdamW(self.policyNet.parameters(),
                                     lr = self.lr, amsgrad=True)

    def act(self, observation:np.ndarray) -> T.Tensor:
        self.epsilon = self.epsEnd + (self.epsStart - self.epsEnd)*\
            math.exp(-1. * self._steps / self.epsDecay)
        self._steps += 1

        if random.random() > self.epsilon:
            state  = T.tensor([observation]).to(self.device)
            action = self.policyNet.forward(state)
        else:
            action = T.tensor(self._env.action_space.sample())
        return action
    
    def memorize(self, state:np.ndarray, action:np.ndarray, 
                 reward:np.float32, nextState:np.ndarray, done:np.bool_):
        self.stateMem.push(state)
        self.nextStateMem.push(nextState)
        self.actionMem.push(action)
        self.rewardMem.push(reward)
        self.doneMem.push(done)
    
    def learn(self):
        pass