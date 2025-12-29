"""Quick test of inference - load directly from checkpoint."""
import os
import sys
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.from_config import from_config
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from gymnasium.spaces import Box

# Create observation and action spaces matching what we use
obs_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

# Model path
TRAINED_MODEL_PATH = os.path.abspath("backend/models/saved_models/best")
print(f"Model path: {TRAINED_MODEL_PATH}")

# Check what's in the checkpoint
import json
checkpoint_dir = TRAINED_MODEL_PATH
metadata_path = os.path.join(checkpoint_dir, "algorithm_state.pkl")
if os.path.exists(metadata_path):
    print(f"Found algorithm_state.pkl")

# Check for rl_module folder
rl_module_path = os.path.join(checkpoint_dir, "learner", "module_state")
if os.path.exists(rl_module_path):
    print(f"Found rl_module at: {rl_module_path}")
    for item in os.listdir(rl_module_path):
        print(f"  - {item}")
        subpath = os.path.join(rl_module_path, item)
        if os.path.isdir(subpath):
            for subitem in os.listdir(subpath):
                print(f"    - {subitem}")

# Alternative: Check for policy checkpoints
policies_path = os.path.join(checkpoint_dir, "policies")
if os.path.exists(policies_path):
    print(f"Found policies at: {policies_path}")
    for item in os.listdir(policies_path):
        print(f"  - {item}")

print(f"\nAll files in checkpoint dir:")
for item in os.listdir(checkpoint_dir):
    print(f"  - {item}")
    subpath = os.path.join(checkpoint_dir, item)
    if os.path.isdir(subpath):
        for subitem in os.listdir(subpath):
            print(f"    - {subitem}")
