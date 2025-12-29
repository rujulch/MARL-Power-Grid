"""
Simplified training script for multi-agent RL with training history logging.
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from environment.smart_grid_env import SmartGridEnv
from environment.config import ENV_CONFIG


class TrainingHistory:
    """Track and save training metrics for visualization."""
    
    def __init__(self, save_path: str, num_agents: int, config: dict):
        self.save_path = save_path
        self.history = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "algorithm": "PPO",
                "num_agents": num_agents,
                "config": config
            },
            "iterations": []
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def add_iteration(self, iteration: int, reward: float, reward_min: float, 
                      reward_max: float, episodes: int, timesteps: int,
                      best_reward: float, agent_rewards: dict, stability: float = 0):
        """Add metrics from a training iteration."""
        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "episode_reward_mean": float(reward),
            "episode_reward_min": float(reward_min),
            "episode_reward_max": float(reward_max),
            "episode_len_mean": 288.0,  # Our episode length
            "episodes_total": int(episodes),
            "timesteps_total": int(timesteps),
            "best_reward_so_far": float(best_reward),
            "policy_rewards": {k: float(v) for k, v in agent_rewards.items()} if agent_rewards else {},
            "avg_stability": float(stability)
        }
        self.history["iterations"].append(iteration_data)
        self.save()  # Save after each iteration
    
    def save(self):
        """Save training history to JSON file."""
        self.history["metadata"]["end_time"] = datetime.now().isoformat()
        self.history["metadata"]["total_iterations"] = len(self.history["iterations"])
        
        if self.history["iterations"]:
            rewards = [it["episode_reward_mean"] for it in self.history["iterations"]]
            self.history["metadata"]["final_reward"] = rewards[-1]
            self.history["metadata"]["best_reward"] = max(rewards)
            self.history["metadata"]["best_iteration"] = rewards.index(max(rewards)) + 1
        
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def env_creator(config):
    """Create environment instance for RLlib."""
    return ParallelPettingZooEnv(SmartGridEnv(**config))


def train(num_iterations=50, use_gpu=True):
    """Train with GPU acceleration and save training history."""
    
    # Initialize Ray with GPU support
    ray.init(ignore_reinit_error=True)
    
    # Register environment
    register_env("smart_grid", env_creator)
    
    # Get spaces
    temp_env = SmartGridEnv(**ENV_CONFIG)
    obs_space = temp_env.observation_space("agent_0")
    act_space = temp_env.action_space("agent_0")
    num_agents = len(temp_env.possible_agents)
    
    print(f"\n{'='*60}")
    print("Smart Grid MARL Training")
    print(f"{'='*60}")
    print(f"Agents: {num_agents}")
    print(f"Obs Space: {obs_space}")
    print(f"Act Space: {act_space}")
    print(f"GPU: {'Yes (GTX 1660 Ti)' if use_gpu else 'No'}")
    print(f"Iterations: {num_iterations}")
    print(f"{'='*60}\n")
    
    # Build config with GPU optimization
    config = PPOConfig()
    config = config.environment("smart_grid", env_config=ENV_CONFIG)
    config = config.framework("torch")
    config = config.resources(
        num_gpus=1 if use_gpu else 0,
        num_cpus_for_main_process=1
    )
    
    # Multi-agent setup
    policies = {f"agent_{i}": (None, obs_space, act_space, {}) for i in range(num_agents)}
    config = config.multi_agent(
        policies=policies,
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
    )
    
    # Training params optimized for faster convergence
    config = config.training(
        lr=3e-4,
        gamma=0.99,
        entropy_coeff=0.01,
        train_batch_size=4000,  # Larger batch for GPU efficiency
        num_epochs=10,
        minibatch_size=256
    )
    
    # Build algorithm
    print("Building PPO algorithm...")
    try:
        algo = config.build()
        print("Algorithm built successfully!\n")
    except Exception as e:
        print(f"Error building algorithm: {e}")
        ray.shutdown()
        return
    
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("backend/models/saved_models", exist_ok=True)
    
    # Initialize training history logger
    history_path = os.path.abspath("backend/models/training_history.json")
    training_history = TrainingHistory(
        save_path=history_path,
        num_agents=num_agents,
        config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "entropy_coeff": 0.01,
            "train_batch_size": 4000,
            "num_epochs": 10
        }
    )
    print(f"Training history will be saved to: {history_path}\n")
    
    # Training loop
    best_reward = float('-inf')
    
    for i in range(1, num_iterations + 1):
        try:
            result = algo.train()
            
            # Extract metrics from new API structure
            env_runners = result.get("env_runners", {})
            reward = env_runners.get("episode_return_mean", 0)
            reward_min = env_runners.get("episode_return_min", reward)
            reward_max = env_runners.get("episode_return_max", reward)
            episodes = env_runners.get("num_episodes", 0)
            timesteps = result.get("num_env_steps_sampled_lifetime", 0)
            
            # Get per-agent rewards
            agent_rewards = env_runners.get("agent_episode_returns_mean", {})
            
            # Update best reward
            if reward > best_reward and reward != 0:
                best_reward = reward
            
            # Log to training history
            training_history.add_iteration(
                iteration=i,
                reward=reward,
                reward_min=reward_min,
                reward_max=reward_max,
                episodes=episodes,
                timesteps=timesteps,
                best_reward=best_reward,
                agent_rewards=agent_rewards
            )
            
            # Print progress
            print(f"Iteration {i}/{num_iterations}")
            print(f"  Episode Return: {reward:.2f} (best: {best_reward:.2f})")
            print(f"  Episodes: {episodes}")
            print(f"  Timesteps: {timesteps}")
            if agent_rewards:
                print(f"  Agent Rewards: {', '.join([f'{k}: {v:.1f}' for k, v in agent_rewards.items()])}")
            
            # Save best model
            if reward == best_reward and reward != 0:
                checkpoint = algo.save(os.path.abspath("backend/models/saved_models/best"))
                print(f"  NEW BEST! Saved to: {checkpoint}")
            
            print()
            
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            continue
    
    # Final save
    try:
        final = algo.save(os.path.abspath("results/final"))
        print(f"\nTraining complete! Final model: {final}")
        print(f"Training history saved to: {history_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")
    
    algo.stop()
    ray.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    
    train(num_iterations=args.iterations, use_gpu=not args.no_gpu)
