"""
Training script for multi-agent reinforcement learning on smart grid.

Uses Ray RLlib with PPO for training multiple independent agents
that learn to cooperate through shared environment rewards.
"""

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from environment.smart_grid_env import SmartGridEnv
from environment.config import ENV_CONFIG, TRAINING_CONFIG, MODEL_CONFIG


class TrainingHistory:
    """Track and save training metrics over iterations."""
    
    def __init__(self, save_path: str = "backend/models/training_history.json"):
        self.save_path = save_path
        self.history = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "algorithm": "PPO",
                "num_agents": ENV_CONFIG["num_agents"],
                "config": {
                    "train_batch_size": TRAINING_CONFIG["train_batch_size"],
                    "learning_rate": TRAINING_CONFIG["learning_rate"],
                    "gamma": TRAINING_CONFIG["gamma"]
                }
            },
            "iterations": []
        }
    
    def add_iteration(self, iteration: int, result: dict, best_reward: float):
        """Add metrics from a training iteration."""
        # Extract per-policy rewards if available
        policy_rewards = {}
        try:
            if "policy_reward_mean" in result:
                policy_rewards = {k: float(v) for k, v in result["policy_reward_mean"].items()}
            elif "env_runners" in result and "policy_reward_mean" in result.get("env_runners", {}):
                policy_rewards = {k: float(v) for k, v in result["env_runners"]["policy_reward_mean"].items()}
        except:
            pass
        
        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "episode_reward_mean": float(result.get("episode_reward_mean", 0)),
            "episode_reward_min": float(result.get("episode_reward_min", 0)),
            "episode_reward_max": float(result.get("episode_reward_max", 0)),
            "episode_len_mean": float(result.get("episode_len_mean", 0)),
            "episodes_total": int(result.get("episodes_total", 0)),
            "timesteps_total": int(result.get("timesteps_total", 0)),
            "best_reward_so_far": float(best_reward),
            "policy_rewards": policy_rewards,
            "info": {
                "num_healthy_workers": result.get("num_healthy_workers", 0),
                "num_env_steps_sampled": result.get("num_env_steps_sampled", 0)
            }
        }
        
        self.history["iterations"].append(iteration_data)
    
    def save(self):
        """Save training history to JSON file."""
        self.history["metadata"]["end_time"] = datetime.now().isoformat()
        self.history["metadata"]["total_iterations"] = len(self.history["iterations"])
        
        if self.history["iterations"]:
            final = self.history["iterations"][-1]
            best = max(self.history["iterations"], key=lambda x: x["episode_reward_mean"])
            self.history["metadata"]["final_reward"] = final["episode_reward_mean"]
            self.history["metadata"]["best_reward"] = best["episode_reward_mean"]
            self.history["metadata"]["best_iteration"] = best["iteration"]
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining history saved to: {self.save_path}")


def env_creator(config):
    """Create environment instance for RLlib."""
    return ParallelPettingZooEnv(SmartGridEnv(**config))


def train_smart_grid_agents(
    num_iterations: int = 500,
    checkpoint_freq: int = 10,
    use_gpu: bool = True,
    results_dir: str = "results",
    share_parameters: bool = False
):
    """
    Train multi-agent PPO on smart grid environment.
    
    Args:
        num_iterations: Number of training iterations
        checkpoint_freq: Save checkpoint every N iterations
        use_gpu: Use GPU for training (recommended)
        results_dir: Directory to save results and checkpoints
        share_parameters: If True, all agents share one neural network (parameter sharing)
                         This can improve sample efficiency for homogeneous agents.
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_gpus=1 if use_gpu else 0)
    
    # Register custom environment
    register_env("smart_grid", env_creator)
    
    # Get environment to extract spaces
    temp_env = SmartGridEnv(**ENV_CONFIG)
    obs_space = temp_env.observation_space("agent_0")
    act_space = temp_env.action_space("agent_0")
    
    # Configure PPO for multi-agent with NEW API (Ray 2.52.1)
    config = (
        PPOConfig()
        .environment(
            "smart_grid",
            env_config=ENV_CONFIG
        )
        .framework("torch")
        .resources(
            num_gpus=1 if use_gpu else 0,
            num_cpus_per_learner_worker=1
        )
        .env_runners(
            num_env_runners=TRAINING_CONFIG["num_rollout_workers"],
            num_cpus_per_env_runner=1
        )
        .training(
            train_batch_size_per_learner=TRAINING_CONFIG["train_batch_size"] // max(1, (1 if use_gpu else 0)),
            minibatch_size=TRAINING_CONFIG["sgd_minibatch_size"],
            num_epochs=TRAINING_CONFIG["num_sgd_iter"],
            lr=TRAINING_CONFIG["learning_rate"],
            gamma=TRAINING_CONFIG["gamma"],
            gae_lambda=TRAINING_CONFIG["lambda_"],
            vf_clip_param=10.0,
            entropy_coeff=TRAINING_CONFIG["entropy_coeff"],
            vf_loss_coeff=0.5
        )
    )
    
    # Configure multi-agent policies
    if share_parameters:
        # Parameter Sharing: All agents use the same neural network
        # This improves sample efficiency for homogeneous agents
        config = config.multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {})
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy"
        )
        print("Using PARAMETER SHARING - all agents share one network")
    else:
        # Independent Learning: Each agent has its own neural network
        config = config.multi_agent(
            policies={
                f"agent_{i}": (None, obs_space, act_space, {})
                for i in range(ENV_CONFIG["num_agents"])
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id
        )
        print("Using INDEPENDENT LEARNING - each agent has its own network")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("backend/models/saved_models", exist_ok=True)
    
    # Initialize training history tracker
    history = TrainingHistory()
    
    # Build algorithm
    print("Building PPO algorithm...")
    algo = config.build()
    
    # Training loop
    best_reward = float('-inf')
    
    print(f"\nStarting training for {num_iterations} iterations...")
    print(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Agents: {ENV_CONFIG['num_agents']}")
    print(f"Workers: {TRAINING_CONFIG['num_rollout_workers']}\n")
    
    for iteration in range(1, num_iterations + 1):
        # Train one iteration
        result = algo.train()
        
        # Extract metrics
        episode_reward_mean = result.get("episode_reward_mean", 0)
        episode_len_mean = result.get("episode_len_mean", 0)
        episodes_total = result.get("episodes_total", 0)
        
        # Print progress
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{num_iterations}")
        print(f"{'='*60}")
        print(f"Mean Reward:        {episode_reward_mean:.2f}")
        print(f"Mean Episode Length: {episode_len_mean:.1f}")
        print(f"Episodes Total:     {episodes_total}")
        print(f"Timesteps Total:    {result.get('timesteps_total', 0)}")
        
        # Save to training history
        history.add_iteration(iteration, result, best_reward)
        
        # Save periodic checkpoints
        if iteration % checkpoint_freq == 0:
            checkpoint_dir = algo.save(os.path.join(results_dir, f"checkpoint_{iteration}"))
            print(f"\n✓ Checkpoint saved: {checkpoint_dir}")
        
        # Save best model
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            best_model_path = os.path.abspath("backend/models/saved_models/best_model")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            best_checkpoint = algo.save(best_model_path)
            print(f"\n✓ New best model! Reward: {best_reward:.2f}")
            print(f"  Saved to: {best_checkpoint}")
        
        # Early stopping if converged well
        if iteration > 200 and episode_reward_mean > 50:
            print(f"\n✓ Good convergence achieved! Reward: {episode_reward_mean:.2f}")
            print("  Consider stopping training if results are satisfactory.")
    
    # Save final model
    final_checkpoint = algo.save(os.path.join(results_dir, "final_model"))
    
    # Save training history
    history.save()
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Final Mean Reward:  {episode_reward_mean:.2f}")
    print(f"Best Reward:        {best_reward:.2f}")
    print(f"Final Checkpoint:   {final_checkpoint}")
    print(f"Best Model:         backend/models/saved_models/best_model")
    print(f"Training History:   backend/models/training_history.json")
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    return final_checkpoint


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Smart Grid MARL Agents")
    parser.add_argument("--iterations", type=int, default=500,
                       help="Number of training iterations")
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                       help="Checkpoint frequency")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU training")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory")
    parser.add_argument("--share-params", action="store_true",
                       help="Enable parameter sharing (all agents share one network)")
    
    args = parser.parse_args()
    
    train_smart_grid_agents(
        num_iterations=args.iterations,
        checkpoint_freq=args.checkpoint_freq,
        use_gpu=not args.no_gpu,
        results_dir=args.results_dir,
        share_parameters=args.share_params
    )





