import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import CEMAgent
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel
from utils import preprocess_obs
from wrappers import GymWrapper, RepeatAction


def save_video_as_gif(frames, output_path='video_prediction.gif'):
    """
    make video with given frames and save as GIF
    """
    plt.figure(figsize=(12, 6))
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        plt.title('Left: Real Observation (GT)' + ' '*15 + 'Right: Model Prediction \n Step %d/%d' % 
                  (i+1, len(frames)), fontsize=12)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=150)
    print(f"Saving video to {output_path}...")
    anim.save(output_path, writer='imagemagick', fps=10)
    print(f"Video saved successfully to {output_path}!")


def main():
    parser = argparse.ArgumentParser(description='Open-loop video prediction with learned model')
    parser.add_argument('dir', type=str, help='log directory or checkpoint directory to load learned model')
    parser.add_argument('--length', type=int, default=50,
                        help='the length of video prediction')
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('-R', '--action-repeat', type=int, default=4)
    parser.add_argument('-H', '--horizon', type=int, default=12)
    parser.add_argument('-I', '--N-iterations', type=int, default=10)
    parser.add_argument('-J', '--N-candidates', type=int, default=1000)
    parser.add_argument('-K', '--N-top-candidates', type=int, default=100)
    parser.add_argument('--output', type=str, default='video_prediction.gif',
                        help='Output filename for the video')
    args = parser.parse_args()

    # Check if the provided directory is a checkpoint directory or log directory
    checkpoint_dir = args.dir
    log_dir = args.dir
    
    # If it's a checkpoint directory (contains checkpoint files), find parent for args.json
    if os.path.basename(checkpoint_dir).startswith('checkpoint_ep'):
        log_dir = os.path.dirname(checkpoint_dir)
        print(f"Detected checkpoint directory: {checkpoint_dir}")
        print(f"Using parent log directory for args.json: {log_dir}")
    else:
        # Check if checkpoint files exist directly in the directory
        if os.path.exists(os.path.join(checkpoint_dir, 'encoder.pth')):
            print(f"Loading checkpoints from: {checkpoint_dir}")
        else:
            # Try to find the latest checkpoint directory
            checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) 
                             if os.path.isdir(os.path.join(checkpoint_dir, d)) 
                             and d.startswith('checkpoint_ep')]
            if checkpoint_dirs:
                # Sort by episode number and use the latest
                checkpoint_dirs.sort(key=lambda x: int(x.split('_ep')[1]))
                checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_dirs[-1])
                print(f"Found checkpoint directory: {checkpoint_dir}")
            else:
                raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    # define environment and apply wrapper
    env = suite.load(args.domain_name, args.task_name)
    env = pixels.Wrapper(env, render_kwargs={'height': 64,
                                             'width': 64,
                                             'camera_id': 0})
    env = GymWrapper(env)
    env = RepeatAction(env, skip=args.action_repeat)

    # define models - load args.json from log directory
    args_json_path = os.path.join(log_dir, 'args.json')
    if not os.path.exists(args_json_path):
        raise ValueError(f"args.json not found at {args_json_path}")
    
    with open(args_json_path, 'r') as f:
        train_args = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(train_args['state_dim'],
                                    env.action_space.shape[0],
                                    train_args['rnn_hidden_dim']).to(device)
    obs_model = ObservationModel(train_args['state_dim'],
                                 train_args['rnn_hidden_dim']).to(device)
    reward_model = RewardModel(train_args['state_dim'],
                               train_args['rnn_hidden_dim']).to(device)

    # load learned parameters from checkpoint directory
    encoder_path = os.path.join(checkpoint_dir, 'encoder.pth')
    rssm_path = os.path.join(checkpoint_dir, 'rssm.pth')
    obs_model_path = os.path.join(checkpoint_dir, 'obs_model.pth')
    reward_model_path = os.path.join(checkpoint_dir, 'reward_model.pth')
    
    if not all(os.path.exists(p) for p in [encoder_path, rssm_path, obs_model_path, reward_model_path]):
        raise ValueError(f"Missing checkpoint files in {checkpoint_dir}")
    
    print(f"Loading checkpoints from: {checkpoint_dir}")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    rssm.load_state_dict(torch.load(rssm_path, map_location=device))
    obs_model.load_state_dict(torch.load(obs_model_path, map_location=device))
    reward_model.load_state_dict(torch.load(reward_model_path, map_location=device))
    print("Checkpoints loaded successfully!")

    # define agent
    cem_agent = CEMAgent(encoder, rssm, reward_model,
                         args.horizon, args.N_iterations,
                         args.N_candidates, args.N_top_candidates)

    # open-loop video prediction
    # select starting point of open-loop prediction randomly
    starting_point = torch.randint(1000 // args.action_repeat - args.length, (1,)).item()
    # interact in environment until starting point and charge context in cem_agent.rnn_hidden
    obs = env.reset()
    for _ in range(starting_point):
        action = cem_agent(obs)
        obs, _, _, _ = env.step(action)

    # preprocess observatin and embed by encoder
    preprocessed_obs = preprocess_obs(obs)
    preprocessed_obs = torch.as_tensor(preprocessed_obs, device=device)
    preprocessed_obs = preprocessed_obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
    with torch.no_grad():
        embedded_obs = encoder(preprocessed_obs)

    # compute state using embedded observation
    # NOTE: after this, state is updated only using prior,
    #       it means model doesn't see observation
    rnn_hidden = cem_agent.rnn_hidden
    state = rssm.posterior(rnn_hidden, embedded_obs).sample()
    frames = []
    for step in range(args.length):
        # action is selected same as training time (closed-loop)
        action = cem_agent(obs)
        obs, _, _, _ = env.step(action)
        
        # Use the agent's updated rnn_hidden after calling cem_agent
        rnn_hidden = cem_agent.rnn_hidden

        # update state and reconstruct observation with same action
        action = torch.as_tensor(action, device=device).unsqueeze(0)
        with torch.no_grad():
            state_prior, rnn_hidden = rssm.prior(state, action, rnn_hidden)
            state = state_prior.sample()
            predicted_obs = obs_model(state, rnn_hidden)

        # arrange GT frame and predicted frame in parallel
        real_obs = preprocess_obs(obs)
        pred_obs = predicted_obs.squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()
        
        # Denormalize: convert from [-0.5, 0.5] back to [0, 1] for display
        real_obs_display = (real_obs + 0.5).clip(0.0, 1.0)
        pred_obs_display = (pred_obs + 0.5).clip(0.0, 1.0)
        
        # Create a NEW frame array for each iteration
        frame = np.zeros((64, 128, 3))
        frame[:, :64, :] = real_obs_display
        frame[:, 64:, :] = pred_obs_display
        frames.append(frame.copy())  # Use .copy() to ensure we store a copy, not a reference

    save_video_as_gif(frames, args.output)

if __name__ == '__main__':
    main()
