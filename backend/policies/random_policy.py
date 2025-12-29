"""
Random policy for Smart Grid environment.

This serves as a baseline to compare against trained and heuristic policies.
Actions are sampled uniformly from the action space.
"""

import numpy as np
from typing import Dict, Any


class RandomPolicy:
    """
    Random policy that samples actions uniformly from action space.
    
    Action space: [-1, 1] for each dimension
    - [0:num_agents]: Transfer to each agent
    - [num_agents]: Grid trade
    """
    
    def __init__(self, num_agents: int = 5):
        self.num_agents = num_agents
        self.action_dim = num_agents + 1
    
    def compute_action(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Sample random action.
        
        Args:
            observation: Not used, but kept for API consistency
        
        Returns:
            Random action array in range [-1, 1]
        """
        return np.random.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)


def get_random_action(
    observations: Dict[str, np.ndarray],
    num_agents: int = 5
) -> Dict[str, np.ndarray]:
    """
    Get random actions for all agents.
    
    Args:
        observations: Dict mapping agent_id to observation array
        num_agents: Number of agents
    
    Returns:
        Dict mapping agent_id to random action array
    """
    policy = RandomPolicy(num_agents=num_agents)
    actions = {}
    
    for agent_id in observations.keys():
        actions[agent_id] = policy.compute_action()
    
    return actions

