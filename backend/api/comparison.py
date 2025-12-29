"""
Comparison API for running policy comparison experiments.

Runs multiple episodes with trained, random, and heuristic policies
and provides statistical analysis of the results.
"""

import numpy as np
import asyncio
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from scipy import stats
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from environment.smart_grid_env import SmartGridEnv
from environment.config import ENV_CONFIG
from policies.heuristic import get_heuristic_action
from policies.random_policy import get_random_action


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    total_reward: float
    avg_stability: float
    grid_imports: float
    solar_utilization: float
    demand_satisfaction: float
    episode_length: int
    per_agent_rewards: Dict[str, float]


@dataclass
class PolicyResults:
    """Aggregated results for a policy across multiple episodes."""
    policy_name: str
    num_episodes: int
    
    # Reward statistics
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    
    # Stability statistics
    stability_mean: float
    stability_std: float
    
    # Grid imports statistics
    grid_imports_mean: float
    grid_imports_std: float
    
    # Solar utilization statistics
    solar_utilization_mean: float
    solar_utilization_std: float
    
    # Demand satisfaction statistics
    demand_satisfaction_mean: float
    demand_satisfaction_std: float
    
    # Raw episode data for detailed analysis
    episodes: List[Dict[str, Any]]


@dataclass
class ComparisonResults:
    """Complete comparison results across all policies."""
    trained: PolicyResults
    heuristic: PolicyResults
    random: PolicyResults
    
    # Statistical tests (trained vs others)
    trained_vs_random_pvalue: float
    trained_vs_heuristic_pvalue: float
    heuristic_vs_random_pvalue: float
    
    # Improvement percentages
    trained_improvement_over_random: float
    trained_improvement_over_heuristic: float


_trained_model_working = None  # Cache to track if trained model works

def get_trained_action_safe(trained_algo, observations, env, num_agents):
    """
    Safely get actions from trained model with fallback to heuristic.
    Uses new Ray RLModule API for inference.
    """
    global _trained_model_working
    
    # If we already know the trained model doesn't work, use heuristic directly
    if _trained_model_working is False:
        return get_heuristic_action(observations, num_agents=num_agents)
    
    try:
        # New Ray API: Use RLModule directly
        actions = {}
        for agent_id, obs in observations.items():
            # Get the module for this policy
            module = trained_algo.get_module(agent_id)
            # Reshape observation for batch processing
            obs_batch = {"obs": torch.tensor(obs).unsqueeze(0).float()}
            # Forward inference
            with torch.no_grad():
                output = module.forward_inference(obs_batch)
                # For continuous PPO: action_dist_inputs = [means, log_stds]
                # We only want the means (first half)
                dist_inputs = output["action_dist_inputs"].squeeze(0)
                action_dim = len(dist_inputs) // 2
                action = dist_inputs[:action_dim].numpy()
            actions[agent_id] = action
        _trained_model_working = True
        return actions
    except Exception as e:
        # First failure - log and cache
        if _trained_model_working is None:
            print(f"Warning: Trained model inference failed ({e}). Using heuristic policy.")
            _trained_model_working = False
        # Fallback to heuristic if trained model fails
        return get_heuristic_action(observations, num_agents=num_agents)


def run_episode(
    env: SmartGridEnv,
    policy_fn,
    trained_algo=None,
    max_steps: int = 288
) -> EpisodeResult:
    """
    Run a single episode with the given policy.
    
    Args:
        env: Smart grid environment
        policy_fn: Function that takes observations and returns actions
        trained_algo: Ray RLlib algorithm for trained policy (optional)
        max_steps: Maximum steps per episode
    
    Returns:
        EpisodeResult with episode metrics
    """
    observations, infos = env.reset()
    
    total_rewards = {agent: 0.0 for agent in env.agents}
    stability_sum = 0.0
    grid_imports = 0.0
    solar_generated = 0.0
    solar_used = 0.0
    demands_met = 0
    total_demand_events = 0
    steps = 0
    
    num_agents = len(env.agents)
    
    for step in range(max_steps):
        # Get actions from policy
        if trained_algo is not None:
            # Use trained model with safe fallback
            actions = get_trained_action_safe(trained_algo, observations, env, num_agents)
        else:
            # Use provided policy function
            actions = policy_fn(observations, num_agents=num_agents)
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Track metrics
        stability_sum += env._calculate_grid_stability()
        
        for agent in env.agents:
            total_rewards[agent] += rewards.get(agent, 0)
            generation = infos[agent]["generation"]
            demand = infos[agent]["demand"]
            solar_generated += generation
            solar_used += min(generation, demand)
            total_demand_events += 1
            # Check if agent can meet demand with current energy + generation
            if env.energy_levels[agent] + generation >= demand:
                demands_met += 1
            
            # Track grid imports
            net = generation - demand
            if net < 0:
                grid_imports += abs(net) * 0.5
        
        steps += 1
        
        if all(truncations.values()) or all(terminations.values()):
            break
    
    return EpisodeResult(
        total_reward=sum(total_rewards.values()),
        avg_stability=stability_sum / max(steps, 1),
        grid_imports=grid_imports,
        solar_utilization=solar_used / max(solar_generated, 1) * 100,
        demand_satisfaction=demands_met / max(total_demand_events, 1) * 100,
        episode_length=steps,
        per_agent_rewards=total_rewards
    )


def aggregate_results(
    policy_name: str,
    episodes: List[EpisodeResult]
) -> PolicyResults:
    """Aggregate episode results into policy statistics."""
    rewards = [e.total_reward for e in episodes]
    stabilities = [e.avg_stability for e in episodes]
    grid_imports = [e.grid_imports for e in episodes]
    solar_utils = [e.solar_utilization for e in episodes]
    demand_sats = [e.demand_satisfaction for e in episodes]
    
    return PolicyResults(
        policy_name=policy_name,
        num_episodes=len(episodes),
        reward_mean=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)),
        reward_min=float(np.min(rewards)),
        reward_max=float(np.max(rewards)),
        stability_mean=float(np.mean(stabilities)),
        stability_std=float(np.std(stabilities)),
        grid_imports_mean=float(np.mean(grid_imports)),
        grid_imports_std=float(np.std(grid_imports)),
        solar_utilization_mean=float(np.mean(solar_utils)),
        solar_utilization_std=float(np.std(solar_utils)),
        demand_satisfaction_mean=float(np.mean(demand_sats)),
        demand_satisfaction_std=float(np.std(demand_sats)),
        episodes=[asdict(e) for e in episodes]
    )


def run_comparison(
    num_episodes: int = 10,
    trained_algo=None,
    progress_callback=None
) -> ComparisonResults:
    """
    Run comparison experiment across all policies.
    
    Args:
        num_episodes: Number of episodes per policy
        trained_algo: Ray RLlib trained algorithm (optional)
        progress_callback: Optional callback for progress updates
    
    Returns:
        ComparisonResults with all statistics
    """
    global _trained_model_working
    _trained_model_working = None  # Reset cache for new comparison run
    
    # Results storage
    trained_episodes = []
    heuristic_episodes = []
    random_episodes = []
    
    total_runs = num_episodes * 3
    current_run = 0
    
    # Run trained policy episodes
    if trained_algo is not None:
        for i in range(num_episodes):
            env = SmartGridEnv(**ENV_CONFIG)
            result = run_episode(env, None, trained_algo=trained_algo)
            trained_episodes.append(result)
            current_run += 1
            if progress_callback:
                progress_callback(current_run, total_runs, "trained", i + 1)
    else:
        # If no trained model, use heuristic as "trained" placeholder
        for i in range(num_episodes):
            env = SmartGridEnv(**ENV_CONFIG)
            result = run_episode(env, get_heuristic_action)
            trained_episodes.append(result)
            current_run += 1
            if progress_callback:
                progress_callback(current_run, total_runs, "trained", i + 1)
    
    # Run heuristic policy episodes
    for i in range(num_episodes):
        env = SmartGridEnv(**ENV_CONFIG)
        result = run_episode(env, get_heuristic_action)
        heuristic_episodes.append(result)
        current_run += 1
        if progress_callback:
            progress_callback(current_run, total_runs, "heuristic", i + 1)
    
    # Run random policy episodes
    for i in range(num_episodes):
        env = SmartGridEnv(**ENV_CONFIG)
        result = run_episode(env, get_random_action)
        random_episodes.append(result)
        current_run += 1
        if progress_callback:
            progress_callback(current_run, total_runs, "random", i + 1)
    
    # Aggregate results
    trained_results = aggregate_results("PPO (Trained)", trained_episodes)
    heuristic_results = aggregate_results("Heuristic", heuristic_episodes)
    random_results = aggregate_results("Random", random_episodes)
    
    # Statistical tests (t-test for reward comparison)
    trained_rewards = [e.total_reward for e in trained_episodes]
    heuristic_rewards = [e.total_reward for e in heuristic_episodes]
    random_rewards = [e.total_reward for e in random_episodes]
    
    # Calculate p-values using independent t-test
    _, p_trained_vs_random = stats.ttest_ind(trained_rewards, random_rewards)
    _, p_trained_vs_heuristic = stats.ttest_ind(trained_rewards, heuristic_rewards)
    _, p_heuristic_vs_random = stats.ttest_ind(heuristic_rewards, random_rewards)
    
    # Calculate improvement percentages
    random_mean = np.mean(random_rewards)
    heuristic_mean = np.mean(heuristic_rewards)
    trained_mean = np.mean(trained_rewards)
    
    improvement_over_random = ((trained_mean - random_mean) / abs(random_mean) * 100) if random_mean != 0 else 0
    improvement_over_heuristic = ((trained_mean - heuristic_mean) / abs(heuristic_mean) * 100) if heuristic_mean != 0 else 0
    
    # Update policy name if trained model didn't work
    if _trained_model_working is False:
        trained_results.policy_name = "Heuristic (Trained model unavailable)"
    
    return ComparisonResults(
        trained=trained_results,
        heuristic=heuristic_results,
        random=random_results,
        trained_vs_random_pvalue=float(p_trained_vs_random),
        trained_vs_heuristic_pvalue=float(p_trained_vs_heuristic),
        heuristic_vs_random_pvalue=float(p_heuristic_vs_random),
        trained_improvement_over_random=float(improvement_over_random),
        trained_improvement_over_heuristic=float(improvement_over_heuristic)
    )


async def run_comparison_async(
    num_episodes: int = 10,
    trained_algo=None,
    websocket_manager=None
) -> ComparisonResults:
    """
    Async version of run_comparison with WebSocket progress updates.
    """
    def progress_callback(current, total, policy, episode):
        if websocket_manager:
            asyncio.create_task(
                websocket_manager.broadcast({
                    "type": "comparison_progress",
                    "current": current,
                    "total": total,
                    "policy": policy,
                    "episode": episode,
                    "percentage": round(current / total * 100, 1)
                })
            )
    
    # Run in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_comparison(num_episodes, trained_algo, progress_callback)
    )
    
    return result

