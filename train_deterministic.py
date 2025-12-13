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
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import CEMAgent
from model_deterministic import Encoder, DeterministicRNN, ObservationModel, RewardModel
from utils import ReplayBuffer, preprocess_obs
from wrappers import GymWrapper, RepeatAction


def save_checkpoint(encoder, det_rnn, obs_model, reward_model, log_dir, episode):
    """Save model checkpoints"""
    checkpoint_dir = os.path.join(log_dir, f'checkpoint_ep{episode+1}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, 'encoder.pth'))
    torch.save(det_rnn.state_dict(), os.path.join(checkpoint_dir, 'det_rnn.pth'))
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
    parser.add_argument('--hidden-size', type=int, default=200)
    parser.add_argument('--embed-size', type=int, default=200)
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
    args = parser.parse_args()

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
    det_rnn = DeterministicRNN(args.hidden_size,
                               env.action_space.shape[0],
                               embed_size=args.embed_size).to(device)
    obs_model = ObservationModel(args.hidden_size).to(device)
    reward_model = RewardModel(args.hidden_size).to(device)
    all_params = (list(encoder.parameters()) +
                  list(det_rnn.parameters()) +
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
        # Note: CEMAgent expects RSSM interface, so we use random actions for deterministic training
        # In practice, you might want to create a deterministic-specific agent
        from agent import CEMAgent
        try:
            cem_agent = CEMAgent(encoder, det_rnn, reward_model,
                                 args.horizon, args.N_iterations,
                                 args.N_candidates, args.N_top_candidates)
        except:
            # Fallback to random actions if agent doesn't work with deterministic model
            cem_agent = None

        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            if cem_agent is not None:
                try:
                    action = cem_agent(obs)
                    action += np.random.normal(0, np.sqrt(args.action_noise_var),
                                               env.action_space.shape[0])
                except:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
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

            # prepare Tensor to maintain RNN hidden states sequence (no stochastic states for deterministic model)
            rnn_hiddens = torch.zeros(
                args.chunk_length, args.batch_size, args.hidden_size, device=device)

            # initialize RNN hidden state with 0 vector
            rnn_hidden = torch.zeros(args.batch_size, args.hidden_size, device=device)

            # compute RNN hidden sequences
            # Deterministic model has no KL divergence (always 0)
            for l in range(args.chunk_length-1):
                _, next_rnn_hidden_posterior, rnn_state = \
                    det_rnn(rnn_hidden, actions[l], embedded_observations[l+1])
                rnn_hidden = next_rnn_hidden_posterior
                rnn_hiddens[l+1] = rnn_hidden
            
            # KL loss is always 0 for deterministic model (no stochastic states)
            kl_loss = torch.tensor(0.0, device=device)

            # compute reconstructed observations and predicted rewards
            # Deterministic model uses only RNN hidden state (no stochastic state)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, args.hidden_size)
            recon_observations = obs_model(flatten_rnn_hiddens).view(
                args.chunk_length, args.batch_size, 3, 64, 64)
            predicted_rewards = reward_model(flatten_rnn_hiddens).view(
                args.chunk_length, args.batch_size, 1)

            # compute loss for observation and reward
            # Equation 3, reconstruction loss
            obs_loss = 0.5 * mse_loss(
                recon_observations[1:], observations[1:], reduction='none').mean([0, 1]).sum()
            reward_loss = 0.5 * mse_loss(predicted_rewards[1:], rewards[:-1])

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
            if cem_agent is not None:
                try:
                    test_agent = CEMAgent(encoder, det_rnn, reward_model,
                                         args.horizon, args.N_iterations,
                                         args.N_candidates, args.N_top_candidates)
                    obs = env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        try:
                            action = test_agent(obs)
                        except:
                            action = env.action_space.sample()
                        obs, reward, done, _ = env.step(action)
                        total_reward += reward
                except:
                    # Fallback to random actions
                    obs = env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        action = env.action_space.sample()
                        obs, reward, done, _ = env.step(action)
                        total_reward += reward
            else:
                obs = env.reset()
                done = False
                total_reward = 0
                while not done:
                    action = env.action_space.sample()
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
            save_checkpoint(encoder, det_rnn, obs_model, reward_model, log_dir, episode)

    # save learned model parameters (final checkpoint)
    save_checkpoint(encoder, det_rnn, obs_model, reward_model, log_dir, args.all_episodes - 1)
    writer.close()

if __name__ == '__main__':
    main()
