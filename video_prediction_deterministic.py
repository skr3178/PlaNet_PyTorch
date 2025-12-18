import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels
from model_deterministic import Encoder, DeterministicRNN, ObservationModel, RewardModel
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
    parser = argparse.ArgumentParser(description='Open-loop video prediction with learned deterministic model')
    parser.add_argument('dir', type=str, help='log directory or checkpoint directory to load learned model')
    parser.add_argument('--length', type=int, default=50,
                        help='the length of video prediction')
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('-R', '--action-repeat', type=int, default=4)
    parser.add_argument('--output', type=str, default='video_prediction_deterministic.gif',
                        help='Output filename for the video')
    parser.add_argument('--action-noise', type=float, default=0.3,
                        help='Action noise variance for exploration')
    args = parser.parse_args()

    # Normalize the path (handle relative paths correctly)
    checkpoint_dir = args.dir
    # If path doesn't exist, try removing leading PlaNet_PyTorch/ if present
    if not os.path.exists(checkpoint_dir) and checkpoint_dir.startswith('PlaNet_PyTorch/'):
        # Try without the prefix
        alt_path = checkpoint_dir[len('PlaNet_PyTorch/'):]
        if os.path.exists(alt_path):
            checkpoint_dir = alt_path
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    log_dir = checkpoint_dir
    
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
    det_rnn = DeterministicRNN(train_args['hidden_size'],
                               env.action_space.shape[0],
                               embed_size=train_args['embed_size']).to(device)
    obs_model = ObservationModel(train_args['hidden_size']).to(device)
    reward_model = RewardModel(train_args['hidden_size']).to(device)

    # load learned parameters from checkpoint directory
    encoder_path = os.path.join(checkpoint_dir, 'encoder.pth')
    det_rnn_path = os.path.join(checkpoint_dir, 'det_rnn.pth')
    obs_model_path = os.path.join(checkpoint_dir, 'obs_model.pth')
    reward_model_path = os.path.join(checkpoint_dir, 'reward_model.pth')
    
    if not all(os.path.exists(p) for p in [encoder_path, det_rnn_path, obs_model_path, reward_model_path]):
        raise ValueError(f"Missing checkpoint files in {checkpoint_dir}")
    
    print(f"Loading checkpoints from: {checkpoint_dir}")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    det_rnn.load_state_dict(torch.load(det_rnn_path, map_location=device))
    obs_model.load_state_dict(torch.load(obs_model_path, map_location=device))
    reward_model.load_state_dict(torch.load(reward_model_path, map_location=device))
    print("Checkpoints loaded successfully!")

    # open-loop video prediction
    # select starting point of open-loop prediction randomly
    starting_point = torch.randint(1000 // args.action_repeat - args.length, (1,)).item()
    
    # interact in environment until starting point to build up context
    obs = env.reset()
    rnn_hidden = torch.zeros(1, train_args['hidden_size'], device=device)
    
    for _ in range(starting_point):
        # Use random actions with noise for exploration
        action = env.action_space.sample()
        action += np.random.normal(0, np.sqrt(args.action_noise), env.action_space.shape[0])
        obs, _, _, _ = env.step(action)
        
        # Update RNN hidden state with actual observations (closed-loop)
        preprocessed_obs = preprocess_obs(obs)
        preprocessed_obs = torch.as_tensor(preprocessed_obs, device=device)
        preprocessed_obs = preprocessed_obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            embedded_obs = encoder(preprocessed_obs)
            action_tensor = torch.as_tensor(action, device=device).unsqueeze(0)
            _, rnn_hidden, _ = det_rnn.forward(rnn_hidden, action_tensor, embedded_obs)
            # rnn_hidden now contains the posterior hidden state

    # preprocess observation and embed by encoder
    preprocessed_obs = preprocess_obs(obs)
    preprocessed_obs = torch.as_tensor(preprocessed_obs, device=device)
    preprocessed_obs = preprocessed_obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
    with torch.no_grad():
        embedded_obs = encoder(preprocessed_obs)
        # Initialize RNN hidden state from current observation (posterior)
        _, rnn_hidden, _ = det_rnn.forward(rnn_hidden, 
                                           torch.zeros(1, env.action_space.shape[0], device=device),
                                           embedded_obs)

    frames = []
    for step in range(args.length):
        # action is selected with noise for exploration (similar to training)
        action = env.action_space.sample()
        action += np.random.normal(0, np.sqrt(args.action_noise), env.action_space.shape[0])
        obs, _, _, _ = env.step(action)
        
        # update RNN hidden state using prior (open-loop prediction)
        # NOTE: after this, hidden state is updated only using prior,
        #       it means model doesn't see observation
        action_tensor = torch.as_tensor(action, device=device).unsqueeze(0)
        with torch.no_grad():
            # Use prior to predict next hidden state (open-loop)
            rnn_hidden, _ = det_rnn.prior(rnn_hidden, action_tensor)
            # Reconstruct observation from predicted hidden state
            predicted_obs = obs_model(rnn_hidden)

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
