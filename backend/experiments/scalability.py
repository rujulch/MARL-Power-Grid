"""
Scalability Analysis for Smart Grid MARL

Tests how the system performs with different numbers of agents.
Measures: training time, convergence, reward, memory usage.
"""

import numpy as np
import time
import json
import psutil
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from environment.smart_grid_env import SmartGridEnv
from policies.heuristic import get_heuristic_action
from policies.random_policy import get_random_action


@dataclass
class AgentScaleResult:
    """Results for a specific agent count."""
    num_agents: int
    episodes_run: int
    
    # Performance metrics
    mean_reward: float
    std_reward: float
    mean_stability: float
    std_stability: float
    
    # Timing metrics
    total_time_seconds: float
    avg_episode_time_seconds: float
    avg_step_time_ms: float
    
    # Resource metrics
    memory_usage_mb: float
    
    # Environment details
    observation_dim: int
    action_dim: int


@dataclass
class ScalabilityResults:
    """Complete scalability analysis results."""
    timestamp: str
    agent_counts: List[int]
    episodes_per_count: int
    results: List[AgentScaleResult]
    
    # Analysis
    time_scaling_factor: float  # How time scales with agents
    reward_scaling_factor: float  # How reward changes with agents


def run_single_scale_test(
    num_agents: int,
    num_episodes: int = 5,
    steps_per_episode: int = 288,
    policy: str = "heuristic"
) -> AgentScaleResult:
    """
    Run scalability test for a specific agent count.
    
    Args:
        num_agents: Number of agents to test
        num_episodes: Number of episodes to run
        steps_per_episode: Steps per episode
        policy: Which policy to use ("heuristic" or "random")
    
    Returns:
        AgentScaleResult with metrics
    """
    # Create environment
    env_config = {
        "num_agents": num_agents,
        "max_steps": steps_per_episode,
        "max_energy_capacity": 100.0,
        "max_transfer_rate": 20.0
    }
    env = SmartGridEnv(**env_config)
    
    # Get dimensions
    obs_space = env.observation_space("agent_0")
    act_space = env.action_space("agent_0")
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    
    # Select policy function
    if policy == "heuristic":
        policy_fn = lambda obs: get_heuristic_action(obs, num_agents=num_agents)
    else:
        policy_fn = lambda obs: get_random_action(obs, num_agents=num_agents)
    
    # Run episodes
    rewards = []
    stabilities = []
    step_times = []
    
    # Memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    start_time = time.time()
    
    for ep in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0
        episode_stability = 0
        
        for step in range(steps_per_episode):
            step_start = time.time()
            
            actions = policy_fn(observations)
            observations, rewards_dict, _, truncations, _ = env.step(actions)
            
            step_times.append((time.time() - step_start) * 1000)  # ms
            
            episode_reward += sum(rewards_dict.values())
            episode_stability += env._calculate_grid_stability()
            
            if all(truncations.values()):
                break
        
        rewards.append(episode_reward)
        stabilities.append(episode_stability / steps_per_episode)
    
    total_time = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    return AgentScaleResult(
        num_agents=num_agents,
        episodes_run=num_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_stability=float(np.mean(stabilities)),
        std_stability=float(np.std(stabilities)),
        total_time_seconds=total_time,
        avg_episode_time_seconds=total_time / num_episodes,
        avg_step_time_ms=float(np.mean(step_times)),
        memory_usage_mb=mem_after - mem_before,
        observation_dim=obs_dim,
        action_dim=act_dim
    )


def run_scalability_analysis(
    agent_counts: List[int] = [3, 5, 7, 10, 15],
    episodes_per_count: int = 5,
    save_path: Optional[str] = "backend/models/scalability_results.json"
) -> ScalabilityResults:
    """
    Run full scalability analysis.
    
    Args:
        agent_counts: List of agent counts to test
        episodes_per_count: Episodes to run for each count
        save_path: Path to save results (None to skip saving)
    
    Returns:
        ScalabilityResults with all metrics
    """
    from datetime import datetime
    
    results = []
    
    print(f"\n{'='*60}")
    print("Smart Grid MARL Scalability Analysis")
    print(f"{'='*60}")
    print(f"Testing agent counts: {agent_counts}")
    print(f"Episodes per count: {episodes_per_count}")
    print()
    
    for num_agents in agent_counts:
        print(f"Testing {num_agents} agents...")
        result = run_single_scale_test(
            num_agents=num_agents,
            num_episodes=episodes_per_count
        )
        results.append(result)
        print(f"  Reward: {result.mean_reward:.2f} +/- {result.std_reward:.2f}")
        print(f"  Time: {result.total_time_seconds:.2f}s ({result.avg_step_time_ms:.2f}ms/step)")
        print()
    
    # Calculate scaling factors
    times = [r.avg_episode_time_seconds for r in results]
    agents = [r.num_agents for r in results]
    
    # Linear regression for time scaling
    if len(times) > 1:
        time_slope = np.polyfit(agents, times, 1)[0]
        time_scaling = time_slope / times[0] if times[0] > 0 else 0
    else:
        time_scaling = 0
    
    # Reward scaling
    rewards = [r.mean_reward for r in results]
    if len(rewards) > 1 and rewards[0] != 0:
        reward_slope = np.polyfit(agents, rewards, 1)[0]
        reward_scaling = reward_slope / abs(rewards[0])
    else:
        reward_scaling = 0
    
    scalability = ScalabilityResults(
        timestamp=datetime.now().isoformat(),
        agent_counts=agent_counts,
        episodes_per_count=episodes_per_count,
        results=[asdict(r) for r in results],
        time_scaling_factor=float(time_scaling),
        reward_scaling_factor=float(reward_scaling)
    )
    
    # Save results
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(asdict(scalability), f, indent=2)
        print(f"Results saved to: {save_path}")
    
    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}")
    print(f"Time scaling factor: {time_scaling:.4f}")
    print(f"Reward scaling factor: {reward_scaling:.4f}")
    
    return scalability


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Scalability Analysis")
    parser.add_argument("--agents", type=str, default="3,5,7,10,15",
                       help="Comma-separated agent counts")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Episodes per agent count")
    parser.add_argument("--output", type=str, default="backend/models/scalability_results.json",
                       help="Output path")
    
    args = parser.parse_args()
    agent_counts = [int(x) for x in args.agents.split(',')]
    
    run_scalability_analysis(
        agent_counts=agent_counts,
        episodes_per_count=args.episodes,
        save_path=args.output
    )

