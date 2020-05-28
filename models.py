import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
#torch.manual_seed(999)

def hidden_init(layer):
    """
    Used for parameter initialization
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorCriticNetwork(nn.Module):
    """
    The actor critic network
    The Actor and the Critic Share the same input encoder
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        self.hiddenlayers = 2
        hl_size = 8
        self.actorhl = []
        self.critichl = []

        self.fc1 = nn.Linear(state_dim, int(state_dim/2))
        self.fc2 = nn.Linear(int(state_dim/2), hl_size)

        
        for i in range(self.hiddenlayers):
            self.actorhl.append(nn.Linear(hl_size, hl_size))

        for i in range(self.hiddenlayers):
            self.critichl.append(nn.Linear(hl_size, hl_size))
            
        self.actor_out = nn.Linear(hl_size, action_dim)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.critic_out = nn.Linear(hl_size, 1)
        self.reset_parameters()
        
    def forward(self, state):
        """
        Compute forward pass
        Input: state tensor
        Output: tuple of (clampped action, log probabilities, state values)
        """
        x = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x))

        act = F.relu(self.actorhl[0](x1))
        for i in range(1, len(self.actorhl)):
            act = self.actorhl[i](act)
            act = F.relu(act)

        mean = self.actor_out(act)
            
        dist = torch.distributions.Normal(mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action)


        crit = self.critichl[0](x1)
        for i in range(1, len(self.critichl)):
            crit = self.critichl[i](crit)
            crit = F.relu(crit)
        value = self.critic_out(crit)
        
        return action, log_prob, value
    
    def reset_parameters(self):
        """
        Reset parameters to the initial states
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        for i in range(len(self.critichl)):
            self.critichl[i].weight.data.uniform_(*hidden_init(self.critichl[i]))
        for i in range(len(self.actorhl)):
            self.actorhl[i].weight.data.uniform_(*hidden_init(self.actorhl[i]))

        
        
        self.actor_out.weight.data.uniform_(-3e-3, 3e-3)
        self.critic_out.weight.data.uniform_(-3e-3, 3e-3)

