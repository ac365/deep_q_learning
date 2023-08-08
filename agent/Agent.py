from networks.DQN import DQN as dqn
from agent.Memory import Memory as mem
import gymnasium as gym #FIXME? is this really the best way to do this?

import math
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

class Agent:
    def __init__(self, env:gym.Env, obsDims:list, actDims:list,
                  actSpace:np.ndarray, **hyperparams:dict):
        #member variables
        self._env    = env
        self._steps  = 0
        self._device = T.device("mps" if T.backends.mps.is_available()
                                   else "cpu")
        self.actSpace = actSpace

        #kwargs
        if "MEM_CAPACITY" in hyperparams.keys():
            self.stateMem     = mem(hyperparams["MEM_CAPACITY"],
                                    np.float32, *obsDims)
            self.nextStateMem = mem(hyperparams["MEM_CAPACITY"],
                                    np.float32, *obsDims)
            self.actionMem    = mem(hyperparams["MEM_CAPACITY"],
                                    np.int32, *actDims)
            self.rewardMem    = mem(hyperparams["MEM_CAPACITY"],
                                    np.float32)
            self.doneMem      = mem(hyperparams["MEM_CAPACITY"],np.bool_)
        else:
            self.stateMem     = mem(100000, np.float32, *obsDims)
            self.nextStateMem = mem(100000, np.float32, *obsDims)
            self.actionMem    = mem(100000, np.int32,   *actDims) 
            self.rewardMem    = mem(100000, np.float32) 
            self.doneMem      = mem(100000, np.bool_)
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
        self.policyNet = dqn(obsDims, actSpace.size).to(self._device)
        self.targetNet = dqn(obsDims, actSpace.size).to(self._device)
        self.optimizer = optim.AdamW(self.policyNet.parameters(),
                                     lr = self.lr, amsgrad=True)

    def act(self, observation:np.ndarray) -> np.array:
        #FIXME: action should be 1x2 w. argmax to choose best of actions
        self.epsilon = self.epsEnd + (self.epsStart - self.epsEnd)*\
            math.exp(-1. * self._steps / self.epsDecay)
        self._steps += 1

        if random.random() > self.epsilon:
            state  = T.tensor(observation).to(self._device)
            actInd = T.argmax(self.policyNet.forward(state))
            action = np.int32(self.actSpace[actInd])
        else:
            action = np.int32(self._env.action_space.sample())
        return action
    
    def memorize(self, state:np.ndarray, action:np.ndarray, 
                 reward:np.float32, nextState:np.ndarray, done:np.bool_):
        self.stateMem.push(state)
        self.nextStateMem.push(nextState)
        self.actionMem.push(action)
        self.rewardMem.push(reward)
        self.doneMem.push(done)
    
    def learn(self):
        if self.stateMem.counter >= self.batchSize:
            seed = np.random.randint(0,999)
            
            stateBatch    = self.stateMem.sample(self.batchSize, seed)
            nextStateBatch= self.nextStateMem.sample(self.batchSize,seed)
            actionBatch   = self.actionMem.sample(self.batchSize, seed)
            rewardBatch   = self.rewardMem.sample(self.batchSize, seed)
            doneBatch     = self.doneMem.sample(self.batchSize, seed)
            
            T.Tensor(stateBatch).to(self._device)
            T.Tensor(nextStateBatch).to(self._device)
            T.Tensor(rewardBatch).to(self._device)            
            T.Tensor(doneBatch).to(self._device)

            #FIXME? test if .gather(1,actionBatch) is necessary
            action_ = self.policyNet.forward(stateBatch)
            values_ = self.targetNet.forward(nextStateBatch)
            values_[doneBatch] = 0.0
            reward_ = values_.max(1)[0] * self.gamma + rewardBatch

            criterion = nn.SmoothL1Loss()
            loss = criterion(action_, reward_)
