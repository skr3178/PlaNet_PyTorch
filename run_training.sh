#!/bin/bash
# Run training in background with nohup
cd /home/skr/Downloads/robot_world/PlaNet_PyTorch
source $(conda info --base)/etc/profile.d/conda.sh
conda activate planet_pytorch
nohup python3 train.py "$@" > training_output.log 2>&1 &
echo "Training started in background. PID: $!"
echo "Monitor with: tail -f training_output.log"
echo "Check process: ps aux | grep train.py"
echo "Monitor resources: htop (in another terminal)"


python video_prediction.py log/cheetah_run/20251213_070630/checkpoint_ep400 \
    --length 100 \
    --output my_video.gif \
    --domain-name cheetah \
    --task-name run