import torch.nn    as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self,num_obs,num_act):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_obs,128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,num_act)

    def forward(self, x):
        action = F.relu(self.layer1(x))
        action = F.relu(self.layer2(action))
        action = self.layer3(action)
        return action