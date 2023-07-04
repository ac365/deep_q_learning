import Memory
import networks.DQN as DQN

class Agent:
    def __init__(self,BATCH_SIZE=128,GAMMA=.99,EPS_START=0.9, 
                 EPS_END=0.05,EPS_DECAY=1000,TAU=0.005,LR=1e-4,
                 MEM_CAPACITY=100000) -> None:
        self.lr        = LR
        self.tau       = TAU
        self.gamma     = GAMMA
        self.epsEnd    = EPS_END
        self.epsStart  = EPS_START
        self.epsDecay  = EPS_DECAY
        self.batchSize = BATCH_SIZE

        self.policyNet = DQN()
        self.targetNet = DQN()
        self.memory    = Memory(MEM_CAPACITY)
        
    def act():
        pass