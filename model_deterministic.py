import torch
from torch import nn
from torch.nn import functional as F


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


class DeterministicRNN(nn.Module):
    """
    Deterministic RNN model with no stochastic states.
    
    This is a simple deterministic RNN that processes observations and actions
    through a GRU cell. There are no stochastic latent states - only the
    deterministic hidden state.
    
    Prior:    Posterior:
    (a)       (a,o)
       \         \
        v         v
    [h]->[h]  [h]->[h]
    
    Based on the original TensorFlow implementation in planet/planet/models/deterministic_rnn.py
    """
    def __init__(self, hidden_size, action_dim, embed_size=200,
                 activation=F.elu, num_layers=1):
        super(DeterministicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.act = activation
        
        # Prior network: [h_t, a_t] -> GRU -> h_t+1
        self.fc_prior_layers = nn.ModuleList([
            nn.Linear(hidden_size + action_dim if i == 0 else embed_size, embed_size)
            for i in range(num_layers)
        ])
        self.rnn = nn.GRUCell(embed_size, hidden_size)
        
        # Posterior network: [h_t, a_t, o_t] -> GRU -> h_t+1
        # Note: embedded observation is 1024 dimensions (from encoder)
        self.fc_posterior_layers = nn.ModuleList([
            nn.Linear(hidden_size + action_dim + 1024 if i == 0 else embed_size, embed_size)
            for i in range(num_layers)
        ])

    def forward(self, hidden, action, embedded_next_obs):
        """
        Return prior h_t+1 = f(h_t, a_t) and posterior h_t+1 = f(h_t, a_t, o_t+1)
        for model training
        """
        next_hidden_prior, rnn_state_prior = self.prior(hidden, action)
        next_hidden_posterior, rnn_state_posterior = self.posterior(hidden, action, embedded_next_obs)
        return next_hidden_prior, next_hidden_posterior, rnn_state_posterior

    def prior(self, hidden, action):
        """
        Compute prior h_t+1 = f(h_t, a_t)
        [h_t, a_t] -> FC layers -> GRU -> h_t+1
        """
        inputs = torch.cat([hidden, action], dim=1)
        for fc in self.fc_prior_layers:
            inputs = self.act(fc(inputs))
        
        # GRU cell update
        rnn_state = self.rnn(inputs, hidden)
        return rnn_state, rnn_state

    def posterior(self, hidden, action, embedded_obs):
        """
        Compute posterior h_t+1 = f(h_t, a_t, o_t)
        [h_t, a_t, o_t] -> FC layers -> GRU -> h_t+1
        """
        inputs = torch.cat([hidden, action, embedded_obs], dim=1)
        for fc in self.fc_posterior_layers:
            inputs = self.act(fc(inputs))
        
        # GRU cell update
        rnn_state = self.rnn(inputs, hidden)
        return rnn_state, rnn_state


class ObservationModel(nn.Module):
    """
    p(o_t | h_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from RNN hidden state only (no stochastic state for deterministic model)
    """
    def __init__(self, hidden_size):
        super(ObservationModel, self).__init__()
        self.fc = nn.Linear(hidden_size, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, hidden):
        hidden_state = self.fc(hidden)
        hidden_state = hidden_state.view(hidden_state.size(0), 1024, 1, 1)
        hidden_state = F.relu(self.dc1(hidden_state))
        hidden_state = F.relu(self.dc2(hidden_state))
        hidden_state = F.relu(self.dc3(hidden_state))
        obs = self.dc4(hidden_state)
        return obs


class RewardModel(nn.Module):
    """
    p(r_t | h_t)
    Reward model to predict reward from RNN hidden state only (no stochastic state for deterministic model)
    """
    def __init__(self, hidden_size, hidden_dim=300, act=F.relu):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, hidden):
        hidden_state = self.act(self.fc1(hidden))
        hidden_state = self.act(self.fc2(hidden_state))
        hidden_state = self.act(self.fc3(hidden_state))
        reward = self.fc4(hidden_state)
        return reward
