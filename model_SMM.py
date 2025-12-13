import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class Encoder(nn.Module):
    """
    Encoder to embed image observation (3, 64, 64) to vector (1024,)
    3 channels → 32 → 64 → 128 → 256 channels
    64x64 → 32x32 → 16x16 → 8x8 → 4x4 (spatial dimensions)
    Final: 256 x 4 x 4 = 4096 → reshaped to 1024
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, obs):
        hidden = F.relu(self.cv1(obs))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        embedded_obs = F.relu(self.cv4(hidden)).reshape(hidden.size(0), -1)
        return embedded_obs


class StateSpaceModel(nn.Module):
    """
    Gaussian State Space Model (SSM).
    
    Implements the transition function and encoder using feedforward networks.
    No RNN - just feedforward from state and action.
    
    Prior:    Posterior:
    (a)       (a)
       \         \
        v         v
    (s)->(s)  (s)->(s)
                    ^
                    :
                   (o)
    
    Based on the original TensorFlow implementation in planet/planet/models/ssm.py
    """
    def __init__(self, state_dim, action_dim, embed_size=200,
                 mean_only=False, activation=F.elu, min_stddev=1e-5):
        super(StateSpaceModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_size = embed_size
        self.mean_only = mean_only
        self._min_stddev = min_stddev
        self.act = activation
        
        # Transition network: [s_t, a_t] -> [mean, stddev]
        self.fc_transition = nn.Linear(state_dim + action_dim, embed_size)
        self.fc_mean_prior = nn.Linear(embed_size, state_dim)
        self.fc_stddev_prior = nn.Linear(embed_size, state_dim)
        
        # Posterior network: [prior_mean, prior_stddev, obs] -> [mean, stddev]
        # Note: embedded observation is 1024 dimensions (from encoder)
        self.fc_posterior = nn.Linear(state_dim + state_dim + 1024, embed_size)
        self.fc_mean_posterior = nn.Linear(embed_size, state_dim)
        self.fc_stddev_posterior = nn.Linear(embed_size, state_dim)

    def forward(self, state, action, embedded_next_obs):
        """
        Return prior p(s_t+1 | s_t, a_t) and posterior q(s_t+1 | s_t, a_t, o_t+1)
        for model training
        """
        next_state_prior = self.prior(state, action)
        next_state_posterior = self.posterior(state, action, embedded_next_obs)
        return next_state_prior, next_state_posterior

    def prior(self, state, action):
        """
        Compute prior p(s_t+1 | s_t, a_t)
        [s_t, a_t] -> FC -> activation -> [mean, stddev] -> Normal distribution
        """
        hidden = self.act(self.fc_transition(torch.cat([state, action], dim=1)))
        mean = self.fc_mean_prior(hidden)
        stddev = F.softplus(self.fc_stddev_prior(hidden)) + self._min_stddev
        
        if self.mean_only:
            # Return a dummy distribution with mean only
            return Normal(mean, torch.ones_like(stddev) * 1e-8)
        else:
            return Normal(mean, stddev)

    def posterior(self, prev_state, prev_action, embedded_obs):
        """
        Compute posterior q(s_t | s_t-1, a_t-1, o_t)
        First compute prior, then refine with observation
        [prior_mean, prior_stddev, obs] -> FC -> activation -> [mean, stddev] -> Normal distribution
        """
        # Compute prior first
        prior = self.prior(prev_state, prev_action)
        
        # Use prior mean and stddev along with observation
        inputs = torch.cat([prior.mean, prior.stddev, embedded_obs], dim=1)
        hidden = self.act(self.fc_posterior(inputs))
        mean = self.fc_mean_posterior(hidden)
        stddev = F.softplus(self.fc_stddev_posterior(hidden)) + self._min_stddev
        
        if self.mean_only:
            return Normal(mean, torch.ones_like(stddev) * 1e-8)
        else:
            return Normal(mean, stddev)


class ObservationModel(nn.Module):
    """
    p(o_t | s_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from state only (no RNN hidden state for SSM)
    """
    def __init__(self, state_dim):
        super(ObservationModel, self).__init__()
        self.fc = nn.Linear(state_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, state):
        hidden = self.fc(state)
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs


class RewardModel(nn.Module):
    """
    p(r_t | s_t)
    Reward model to predict reward from state only (no RNN hidden state for SSM)
    """
    def __init__(self, state_dim, hidden_dim=300, act=F.relu):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state):
        hidden = self.act(self.fc1(state))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        reward = self.fc4(hidden)
        return reward
