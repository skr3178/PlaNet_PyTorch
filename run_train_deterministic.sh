#!/bin/bash
# Run Deterministic RNN training in background with nohup
cd /home/skr/Downloads/robot_world/PlaNet_PyTorch
source $(conda info --base)/etc/profile.d/conda.sh
conda activate planet_pytorch
nohup python3 train_deterministic.py "$@" > training_deterministic_output.log 2>&1 &
echo "Deterministic RNN Training started in background. PID: $!"
echo "Monitor with: tail -f training_deterministic_output.log"
echo "Check process: ps aux | grep train_deterministic.py"
echo "Monitor resources: htop (in another terminal)"
echo "Kill process: kill $!"







