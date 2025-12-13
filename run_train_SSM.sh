#!/bin/bash
# Run SSM training in background with nohup
cd /home/skr/Downloads/robot_world/PlaNet_PyTorch
source $(conda info --base)/etc/profile.d/conda.sh
conda activate planet_pytorch
nohup python3 train_SSM.py "$@" > training_ssm_output.log 2>&1 &
echo "SSM Training started in background. PID: $!"
echo "Monitor with: tail -f training_ssm_output.log"
echo "Check process: ps aux | grep train_SSM.py"
echo "Monitor resources: htop (in another terminal)"
echo "Kill process: kill $!"
