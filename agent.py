import torch
from torch.distributions import Normal
from utils import preprocess_obs


class CEMAgent:
    """
    Iteration 1: Sample 1000 random actions → Top 100 → Update distribution
    Iteration 2: Sample 1000 from refined distribution → Top 100 → Update again
    ...
    Iteration 10: Distribution converged to high-reward actions → Return first action
    
    Key Parameters:
    - horizon: Planning horizon (how many steps ahead to plan)
    - N_iterations: Number of CEM iterations to refine the action distribution
    - N_candidates: Number of action sequences to sample per iteration
    - N_top_candidates: Number of top candidates used to update the distribution
    """
    def __init__(self, encoder, rssm, reward_model,
                 horizon, N_iterations, N_candidates, N_top_candidates):
        self.encoder = encoder  # Encodes raw observations to embeddings
        self.rssm = rssm  # Recurrent State Space Model (world model)
        self.reward_model = reward_model  # Predicts rewards from states

        self.horizon = horizon  # Planning horizon (e.g., 12 steps)
        self.N_iterations = N_iterations  # CEM iterations (e.g., 10)
        self.N_candidates = N_candidates  # Samples per iteration (e.g., 1000)
        self.N_top_candidates = N_top_candidates  # Top-K for distribution update (e.g., 100)

        self.device = next(self.reward_model.parameters()).device
        # RNN hidden state tracks deterministic history across planning steps
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs):
        """
        Main planning function: Uses CEM to find the best action sequence.
        
        Algorithm Overview:
        1. Encode current observation and infer posterior state (ground truth state)
        2. Initialize action distribution (Gaussian with mean=0, std=1)
        3. For each CEM iteration:
           a. Sample N_candidates action sequences from current distribution
           b. For each candidate, simulate forward using RSSM prior (open-loop)
           c. Compute total predicted reward over horizon
           d. Select top N_top_candidates based on reward
           e. Update action distribution to match top candidates
        4. Return first action from optimized sequence (receding horizon)
        5. Update RNN hidden state for next planning step
        """
        # Preprocess observation and transpose for torch style (channel-first)
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            # STEP 1: Compute starting state for planning
            # Use posterior (inferred from observation) as ground truth starting state
            # Posterior = q(s_t | h_t, o_t) - state conditioned on observation
            embedded_obs = self.encoder(obs)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)

            # STEP 2: Initialize action distribution
            # Start with zero-mean, unit-variance Gaussian for each timestep
            # This represents "no prior knowledge" about good actions
            action_dist = Normal(
                torch.zeros((self.horizon, self.rssm.action_dim), device=self.device),
                torch.ones((self.horizon, self.rssm.action_dim), device=self.device)
            )

            # Iteratively improve action distribution with CEM
            for itr in range(self.N_iterations):
                # Sample action candidates and transpose to
                # (self.horizon, self.N_candidates, action_dim) for parallel exploration
                action_candidates = \
                    action_dist.sample([self.N_candidates]).transpose(0, 1)

                # Initialize reward, state, and rnn hidden state
                # The size of state is (self.N_acndidates, state_dim)
                # The size of rnn hidden is (self.N_candidates, rnn_hidden_dim)
                # These are for parallel exploration
                total_predicted_reward = torch.zeros(self.N_candidates, device=self.device)
                state = state_posterior.sample([self.N_candidates]).squeeze()
                rnn_hidden = self.rnn_hidden.repeat([self.N_candidates, 1])

                # Compute total predicted reward by open-loop prediction using prior
                for t in range(self.horizon):
                    next_state_prior, rnn_hidden = \
                        self.rssm.prior(state, action_candidates[t], rnn_hidden)
                    state = next_state_prior.sample()
                    total_predicted_reward += self.reward_model(state, rnn_hidden).squeeze()

                # update action distribution using top-k samples
                top_indexes = \
                    total_predicted_reward.argsort(descending=True)[: self.N_top_candidates]
                top_action_candidates = action_candidates[:, top_indexes, :]
                mean = top_action_candidates.mean(dim=1)
                stddev = (top_action_candidates - mean.unsqueeze(1)
                          ).abs().sum(dim=1) / (self.N_top_candidates - 1)
                action_dist = Normal(mean, stddev)

        # Return only first action (replan each state based on new observation)
        action = mean[0]

        # update rnn hidden state for next step planning
        with torch.no_grad():
            _, self.rnn_hidden = self.rssm.prior(state_posterior.sample(),
                                                 action.unsqueeze(0),
                                                 self.rnn_hidden)
        return action.cpu().numpy()

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
