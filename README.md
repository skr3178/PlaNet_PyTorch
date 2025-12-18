# PlaNet_PyTorch
Unofficial re-implementation of "Learning Latent Dynamics for Planning from Pixels" (https://arxiv.org/abs/1811.04551 )

## Instructions
For training, install the requirements (see below) and run (default environment is cheetah run)
```python
python3 train.py
```

For training the deterministic model variant, run
```bash
bash run_train_deterministic.sh
```
or
```python
python3 train_deterministic.py
```

To test learned model, run
```python
python3 test.py dir
```

To predict video with learned model, run
```bash
python3 video_prediction.py log/cheetah_run/20251213_070630
```

To predict video with learned deterministic model, run
```bash
python3 video_prediction_deterministic.py log/cheetah_run/20251217_104737
```

**Video Prediction Options:**
- `dir` - Log directory or checkpoint directory (required)
- `--length` - Length of video prediction in steps (default: 50)
- `--domain-name` - Environment domain (default: cheetah)
- `--task-name` - Environment task (default: run)
- `--action-repeat` - Action repeat factor (default: 4)
- `--output` - Output filename for the GIF (default: `video_prediction.gif` or `video_prediction_deterministic.gif`)
- `--action-noise` - Action noise variance for exploration (default: 0.3)

**Examples:**
```bash
# Generate GIF with PlaNet model (episode 800 checkpoint)
python3 video_prediction.py log/cheetah_run/20251213_070630/checkpoint_ep800 --length 100 --output my_prediction.gif

# Generate GIF with deterministic model (episode 600 checkpoint)
python3 video_prediction_deterministic.py log/cheetah_run/20251217_104737/checkpoint_ep600 --length 100 --output deterministic_prediction.gif
```

The script will automatically find the checkpoint files and generate a GIF showing ground truth observations (left) vs. model predictions (right) side-by-side.

**Note:** The deterministic model has been successfully trained and tested. Both model checkpoints are available in the repository.



## Requirements
* Python 3.9
* Mujoco (for DeepMind Control Suite)

### Installation

1. Create a conda environment:
```bash
conda create -n planet_pytorch python=3.9
conda activate planet_pytorch
```

2. Install mujoco and pyopengl via conda (required for compatibility):
```bash
conda install -c conda-forge mujoco=3.3.2 pyopengl=3.1.10
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Apply compatibility patch for dm_control:
```bash
python3 << 'EOF'
import re

file_path = '/home/skr/miniconda3/envs/planet_pytorch/lib/python3.9/site-packages/dm_control/mujoco/index.py'

with open(file_path, 'r') as f:
    content = f.read()

# Replace the getattr line to handle missing attributes
old_pattern = r'    # Skip over structured arrays and fields that have sizes but aren\'t numpy\n    # arrays, such as text fields and contacts \(b/34805932\)\.\n    attr = getattr\(struct, field_name\)'
new_pattern = r'    # Skip over structured arrays and fields that have sizes but aren\'t numpy\n    # arrays, such as text fields and contacts (b/34805932).\n    # Also skip fields that don\'t exist in this mujoco version\n    try:\n      attr = getattr(struct, field_name)\n    except AttributeError:\n      continue'

content = re.sub(old_pattern, new_pattern, content)

with open(file_path, 'w') as f:
    f.write(content)

print("Patched index.py successfully")
EOF
```

**Note:** Replace `/home/skr/miniconda3/envs/planet_pytorch` with your actual conda environment path.

See `requirements.txt` for exact package versions.

## Qualitative tesult
Example of predicted video frame by learned model
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/video_prediction.gif)

## Quantitative result
### cartpole swingup
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/figures/cartpole_swingup.png)

### reacher easy
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/figures/reacher_easy.png)

### cheetah run
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/figures/cheetah_run.png)

### finger spin
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/figures/finger_spin.png)

### ball_in_cup catch
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/figures/ball_in_cup_catch.png)

### walker walk
![](https://github.com/cross32768/PlaNet_PyTorch/blob/master/figures/walker_walk.png)

Work in progress.

I'm going to add result of experiments at least three times for each environment in the original paper.

All results are test score (without exploration noise), acquired at every 10 episodes.

And I applied moving average with window size=5

## References
* [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
* [Official Implementation](https://github.com/google-research/planet)


## TODO
* speed up training
* Add more qualitative results (at least 3 experiments for each envifonment with different random seed)
* Generalize code for other environments
