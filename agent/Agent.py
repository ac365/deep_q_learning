import Memory
import networks.DQN as DQN

class Agent:
    def __init__(self, obsDims:list, actDims:list, **hyperparams:dict):
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
        
        if "MEM_CAPACITY" in hyperparams.keys():
            self.stateMem  = Memory(hyperparams["MEM_CAPACITY"],*obsDims)
            self.rewardMem = Memory(hyperparams["MEM_CAPACITY"],*obsDims)
            self.actionMem = Memory(hyperparams["MEM_CAPACITY"],*obsDims)
        else:
            self.stateMem  = Memory(100000,*obsDims)
            self.rewardMem = Memory(100000) 
            self.actionMem = Memory(100000,*actDims) 

        self.policyNet = DQN(obsDims,actDims)
        self.targetNet = DQN(obsDims,actDims)

    def act():
        pass