import argparse
from datetime import datetime
import json
import os
from pprint import pprint
import sys
import time
import numpy as np
import torch
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import CEMAgent
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel
from utils import ReplayBuffer, preprocess_obs
from wrappers import GymWrapper, RepeatAction


def save_checkpoint(encoder, rssm, obs_model, reward_model, log_dir, episode):
    """Save model checkpoints"""
    checkpoint_dir = os.path.join(log_dir, f'checkpoint_ep{episode+1}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, 'encoder.pth'))
    torch.save(rssm.state_dict(), os.path.join(checkpoint_dir, 'rssm.pth'))
    torch.save(obs_model.state_dict(), os.path.join(checkpoint_dir, 'obs_model.pth'))
    torch.save(reward_model.state_dict(), os.path.join(checkpoint_dir, 'reward_model.pth'))
    print(f'Checkpoint saved at episode {episode+1} in {checkpoint_dir}')
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='PlaNet for DM control')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--test-interval', type=int, default=10)
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('-R', '--action-repeat', type=int, default=4)
    parser.add_argument('--state-dim', type=int, default=30)
    parser.add_argument('--rnn-hidden-dim', type=int, default=200)
    parser.add_argument('--buffer-capacity', type=int, default=1000000)
    parser.add_argument('--all-episodes', type=int, default=1000)
    parser.add_argument('-S', '--seed-episodes', type=int, default=5)
    parser.add_argument('-C', '--collect-interval', type=int, default=100)
    parser.add_argument('-B', '--batch-size', type=int, default=50)
    parser.add_argument('-L', '--chunk-length', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--clip-grad-norm', type=int, default=1000)
    parser.add_argument('--free-nats', type=int, default=3)
    parser.add_argument('-H', '--horizon', type=int, default=12)
    parser.add_argument('-I', '--N-iterations', type=int, default=10)
    parser.add_argument('-J', '--N-candidates', type=int, default=1000)
    parser.add_argument('-K', '--N-top-candidates', type=int, default=100)
    parser.add_argument('--action-noise-var', type=float, default=0.3)
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--overshooting-distance', type=int, default=0,
                        help='Latent overshooting distance (0 to disable)')
    parser.add_argument('--overshooting-kl-beta', type=float, default=0.0,
                        help='Latent overshooting KL weight (0 to disable)')
    parser.add_argument('--overshooting-reward-scale', type=float, default=0.0,
                        help='Latent overshooting reward prediction weight (0 to disable)')
    args = parser.parse_args()
    
    # Ensure overshooting distance doesn't exceed chunk length
    if args.overshooting_distance > 0:
        args.overshooting_distance = min(args.chunk_length - 1, args.overshooting_distance)

    # Prepare logging
    log_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name)
    # Include seconds in timestamp to reduce collision probability
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, timestamp)
    # Create directory, or use existing one if it already exists
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    pprint(vars(args))
    sys.stdout.flush()
    writer = SummaryWriter(log_dir=log_dir)

    # set seed (NOTE: some randomness is still remaining (e.g. cuDNN's randomness))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # define env and apply wrappers
    env = suite.load(args.domain_name, args.task_name, task_kwargs={'random': args.seed})
    env = pixels.Wrapper(env, render_kwargs={'height': 64,
                                             'width': 64,
                                             'camera_id': 0})
    env = GymWrapper(env)
    env = RepeatAction(env, skip=args.action_repeat)

    # define replay buffer
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                 observation_shape=env.observation_space.shape,
                                 action_dim=env.action_space.shape[0])

    # define models and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(args.state_dim,
                                    env.action_space.shape[0],
                                    args.rnn_hidden_dim).to(device)
    obs_model = ObservationModel(args.state_dim, args.rnn_hidden_dim).to(device)
    reward_model = RewardModel(args.state_dim, args.rnn_hidden_dim).to(device)
    all_params = (list(encoder.parameters()) +
                  list(rssm.parameters()) +
                  list(obs_model.parameters()) +
                  list(reward_model.parameters()))
    optimizer = Adam(all_params, lr=args.lr, eps=args.eps)

    # collect initial experience with random action
    for episode in range(args.seed_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs

    # main training loop
    for episode in range(args.seed_episodes, args.all_episodes):
        # collect experiences
        start = time.time()
        cem_agent = CEMAgent(encoder, rssm, reward_model,
                             args.horizon, args.N_iterations,
                             args.N_candidates, args.N_top_candidates)

        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = cem_agent(obs)
            action += np.random.normal(0, np.sqrt(args.action_noise_var),
                                       env.action_space.shape[0])
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs
            total_reward += reward

        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, args.all_episodes, total_reward))
        sys.stdout.flush()
        print('elasped time for interaction: %.2fs' % (time.time() - start))
        sys.stdout.flush()

        # update model parameters
        start = time.time()
        for update_step in range(args.collect_interval):
            observations, actions, rewards, _ = \
                replay_buffer.sample(args.batch_size, args.chunk_length)

            # preprocess observations and transpose tensor for RNN training
            observations = preprocess_obs(observations)
            observations = torch.as_tensor(observations, device=device)
            observations = observations.transpose(3, 4).transpose(2, 3)
            observations = observations.transpose(0, 1)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)

            # embed observations with CNN
            embedded_observations = encoder(
                observations.reshape(-1, 3, 64, 64)).view(args.chunk_length, args.batch_size, -1)

            # prepare Tensor to maintain states sequence and rnn hidden states sequence
            states = torch.zeros(
                args.chunk_length, args.batch_size, args.state_dim, device=device)
            rnn_hiddens = torch.zeros(
                args.chunk_length, args.batch_size, args.rnn_hidden_dim, device=device)

            # initialize state and rnn hidden state with 0 vector
            state = torch.zeros(args.batch_size, args.state_dim, device=device)
            rnn_hidden = torch.zeros(args.batch_size, args.rnn_hidden_dim, device=device)

            # Store posterior distributions for overshooting (if enabled)
            posterior_means = torch.zeros(
                args.chunk_length, args.batch_size, args.state_dim, device=device)
            posterior_std_devs = torch.zeros(
                args.chunk_length, args.batch_size, args.state_dim, device=device)
            posterior_states = torch.zeros(
                args.chunk_length, args.batch_size, args.state_dim, device=device)
            prior_means_list = []
            prior_std_devs_list = []

            # compute state and rnn hidden sequences and kl loss
            # the KL loss is computed between the prior and posterior of the state
            # - E_{q(s_{t-1}|o_{≤t-1}, a_{<t-1})} [KL[q(s_t | o_{≤t}, a_{<t}) || p(s_t | s_{t-1}, a_{t-1})]] 
            # Equation 3, Lines 184-185: Accumulates the sum over time steps (matching the Σ_{t=1}^T in the equation)
            # Line 186: Divides by (chunk_length - 1) to convert the sum into an average
            kl_loss = 0
            for l in range(args.chunk_length-1):
                next_state_prior, next_state_posterior, rnn_hidden = \
                    rssm(state, actions[l], rnn_hidden, embedded_observations[l+1])
                state = next_state_posterior.rsample()
                states[l+1] = state
                rnn_hiddens[l+1] = rnn_hidden
                
                # Store posterior distributions for overshooting
                if args.overshooting_kl_beta != 0 or args.overshooting_reward_scale != 0:
                    posterior_means[l+1] = next_state_posterior.mean
                    posterior_std_devs[l+1] = next_state_posterior.stddev
                    posterior_states[l+1] = state.detach()  # Detach for overshooting
                    prior_means_list.append(next_state_prior.mean)
                    prior_std_devs_list.append(next_state_prior.stddev)
                
                kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
                kl_loss += kl.clamp(min=args.free_nats).mean()
            kl_loss /= (args.chunk_length - 1) # Normalize by the number of time steps

            # compute reconstructed observations and predicted rewards
            flatten_states = states.view(-1, args.state_dim)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, args.rnn_hidden_dim)
            recon_observations = obs_model(flatten_states, flatten_rnn_hiddens).view(
                args.chunk_length, args.batch_size, 3, 64, 64)
            predicted_rewards = reward_model(flatten_states, flatten_rnn_hiddens).view(
                args.chunk_length, args.batch_size, 1)

            # compute loss for observation and reward
            # Equation 3, reconstruction loss
            obs_loss = 0.5 * mse_loss(
                recon_observations[1:], observations[1:], reduction='none').mean([0, 1]).sum()
            reward_loss = 0.5 * mse_loss(predicted_rewards[1:], rewards[:-1])

            # Calculate latent overshooting objective for t > 0
            if (args.overshooting_kl_beta != 0 or args.overshooting_reward_scale != 0) and args.overshooting_distance > 0:
                overshooting_kl_loss = torch.tensor(0.0, device=device)
                overshooting_reward_loss = torch.tensor(0.0, device=device)
                
                # Process overshooting for each starting time step t
                for t in range(1, args.chunk_length - 1):
                    d = min(t + args.overshooting_distance, args.chunk_length - 1)  # Overshooting distance
                    if d <= t:
                        continue
                    
                    # Perform open-loop rollout from posterior at time t
                    rollout_state = posterior_states[t].detach()
                    rollout_rnn_hidden = rnn_hiddens[t].detach()
                    rollout_length = d - t
                    
                    # Store priors from open-loop rollout
                    rollout_prior_means = []
                    rollout_prior_std_devs = []
                    rollout_rnn_hiddens_seq = [rollout_rnn_hidden]
                    
                    for i in range(rollout_length):
                        # Open-loop: use prior (no observation)
                        next_prior, rollout_rnn_hidden = rssm.prior(
                            rollout_state, actions[t + i], rollout_rnn_hidden)
                        rollout_state = next_prior.rsample()
                        rollout_prior_means.append(next_prior.mean)
                        rollout_prior_std_devs.append(next_prior.stddev)
                        rollout_rnn_hiddens_seq.append(rollout_rnn_hidden)
                    
                    # Stack into tensors: [rollout_length, batch_size, state_dim]
                    prior_means_seq = torch.stack(rollout_prior_means)
                    prior_std_devs_seq = torch.stack(rollout_prior_std_devs)
                    target_means_seq = posterior_means[t+1:d+1].detach()
                    target_std_devs_seq = posterior_std_devs[t+1:d+1].detach()
                    
                    # Compute KL divergence for overshooting
                    if args.overshooting_kl_beta != 0:
                        prior_dist = Normal(prior_means_seq, prior_std_devs_seq)
                        target_dist = Normal(target_means_seq, target_std_devs_seq)
                        
                        # Compute KL: [rollout_length, batch_size]
                        kl_overshooting = kl_divergence(target_dist, prior_dist).sum(dim=-1)
                        # Apply free nats and average
                        kl_overshooting = kl_overshooting.clamp(min=args.free_nats).mean()
                        overshooting_kl_loss = overshooting_kl_loss + kl_overshooting
                    
                    # Compute reward prediction loss for overshooting
                    if args.overshooting_reward_scale != 0:
                        # Get RNN hidden states for reward prediction (exclude the last one)
                        rollout_rnn_hiddens_for_reward = torch.stack(rollout_rnn_hiddens_seq[:-1])
                        
                        # Predict rewards from prior states
                        prior_states_flat = prior_means_seq.view(-1, args.state_dim)
                        rollout_rnn_hiddens_flat = rollout_rnn_hiddens_for_reward.view(-1, args.rnn_hidden_dim)
                        predicted_rewards_overshooting = reward_model(
                            prior_states_flat, rollout_rnn_hiddens_flat)
                        predicted_rewards_overshooting = predicted_rewards_overshooting.view(
                            rollout_length, args.batch_size, 1)
                        
                        # Target rewards
                        target_rewards_seq = rewards[t:d]
                        
                        # Compute MSE loss
                        reward_diff = (predicted_rewards_overshooting - target_rewards_seq) ** 2
                        overshooting_reward_loss = overshooting_reward_loss + reward_diff.mean()
                
                # Normalize overshooting losses
                num_overshooting_steps = max(0, args.chunk_length - 2)
                if num_overshooting_steps > 0 and args.overshooting_distance > 0:
                    overshooting_kl_loss = (1.0 / args.overshooting_distance) * args.overshooting_kl_beta * \
                        overshooting_kl_loss * num_overshooting_steps
                    overshooting_reward_loss = (1.0 / args.overshooting_distance) * args.overshooting_reward_scale * \
                        overshooting_reward_loss * num_overshooting_steps
                
                # Add overshooting losses
                kl_loss = kl_loss + overshooting_kl_loss
                reward_loss = reward_loss + overshooting_reward_loss

            # add all losses and update model parameters with gradient descent
            loss = kl_loss + obs_loss + reward_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, args.clip_grad_norm)
            optimizer.step()

            # print losses and add tensorboard
            print('update_step: %3d loss: %.5f, kl_loss: %.5f, obs_loss: %.5f, reward_loss: % .5f'
                  % (update_step+1,
                     loss.item(), kl_loss.item(), obs_loss.item(), reward_loss.item()))
            sys.stdout.flush()
            total_update_step = episode * args.collect_interval + update_step
            writer.add_scalar('overall loss', loss.item(), total_update_step)
            writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('reward loss', reward_loss.item(), total_update_step)

        print('elasped time for update: %.2fs' % (time.time() - start))
        sys.stdout.flush()

        # test to get score without exploration noise
        if (episode + 1) % args.test_interval == 0:
            start = time.time()
            cem_agent = CEMAgent(encoder, rssm, reward_model,
                                 args.horizon, args.N_iterations,
                                 args.N_candidates, args.N_top_candidates)
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = cem_agent(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            writer.add_scalar('total reward at test', total_reward, episode)
            print('Total test reward at episode [%4d/%4d] is %f' %
                  (episode+1, args.all_episodes, total_reward))
            sys.stdout.flush()
            print('elasped time for test: %.2fs' % (time.time() - start))
            sys.stdout.flush()

        # Save checkpoint periodically
        if (episode + 1) % args.checkpoint_interval == 0:
            save_checkpoint(encoder, rssm, obs_model, reward_model, log_dir, episode)

    # save learned model parameters (final checkpoint)
    save_checkpoint(encoder, rssm, obs_model, reward_model, log_dir, args.all_episodes - 1)
    writer.close()

if __name__ == '__main__':
    main()
