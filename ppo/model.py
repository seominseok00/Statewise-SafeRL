import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class MLPActor(nn.Module):
    def __init__(self, obs_dim, hid_dim, act_dim, activation=F.tanh):
        super(MLPActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, act_dim)
        self.activation = activation

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, obs):
        x = self.activation(self.fc1(obs))
        x = self.activation(self.fc2(x))
        mu = self.fc3(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hid_dim, activation=F.tanh):
        super(MLPCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)
        self.activation = activation

    def forward(self, obs):
        x = self.activation(self.fc1(obs))
        x = self.activation(self.fc2(x))
        v = self.fc3(x)
        return torch.squeeze(v, -1)
    
    
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=64, activation=F.tanh):
        super(MLPActorCritic, self).__init__()

        self.pi = MLPActor(obs_dim, hid_dim, act_dim, activation)

        self.v = MLPCritic(obs_dim, hid_dim, activation)
        self.vc = MLPCritic(obs_dim, hid_dim, activation)

    def step(self, obs):
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32)

            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
class MLPPenalty(nn.Module):
    def __init__(self, obs_dim, hid_dim=64, activation=F.tanh):
        super(MLPPenalty, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)
        self.activation = activation

    def forward(self, obs):
        x = self.activation(self.fc1(obs))
        x = self.activation(self.fc2(x))
        v = self.fc3(x)
        return torch.squeeze(v, -1)