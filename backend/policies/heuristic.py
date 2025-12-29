"""
Heuristic (rule-based) policy for Smart Grid environment.

This serves as a baseline to compare against the trained RL policy.
The heuristic uses simple rules based on domain knowledge:
1. Maintain battery at ~50% capacity for stability
2. Use solar generation directly when available
3. Share energy with neighbors who are critically low
4. Buy from grid only when necessary
"""

import numpy as np
from typing import Dict, Any, List


class HeuristicPolicy:
    """
    Rule-based policy for energy management.
    
    Rules:
    - If battery < 40%: Buy from grid (action = positive grid trade)
    - If battery > 60%: Sell to grid or share with neighbors
    - If neighbor battery < 30%: Share energy with them
    - Otherwise: Maintain current state
    """
    
    def __init__(
        self,
        num_agents: int = 5,
        max_capacity: float = 100.0,
        max_transfer: float = 20.0,
        low_threshold: float = 0.4,
        high_threshold: float = 0.6,
        critical_threshold: float = 0.3
    ):
        self.num_agents = num_agents
        self.max_capacity = max_capacity
        self.max_transfer = max_transfer
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
    
    def compute_action(self, observation: np.ndarray, agent_idx: int) -> np.ndarray:
        """
        Compute action based on heuristic rules.
        
        Observation structure (based on smart_grid_env.py):
        - [0]: Own energy level (normalized 0-1)
        - [1]: Current demand (normalized)
        - [2]: Solar generation (normalized)
        - [3]: Grid stability (0-1)
        - [4]: Current price (0-1)
        - [5]: Time of day (0-1)
        - [6:6+num_agents]: Neighbor energy levels (normalized)
        
        Action structure:
        - [0:num_agents]: Transfer to each agent (-1 to 1, negative = send)
        - [num_agents]: Grid trade (-1 to 1, positive = buy)
        """
        # Parse observation
        own_energy = observation[0]  # Normalized (0-1)
        current_demand = observation[1] * 20.0  # Denormalize
        solar_generation = observation[2] * 15.0  # Denormalize
        grid_stability = observation[3]
        current_price = observation[4]
        time_of_day = observation[5]
        neighbor_energies = observation[6:6 + self.num_agents]
        
        # Initialize action array
        # Actions: [transfer_to_agent_0, ..., transfer_to_agent_n, grid_trade]
        action = np.zeros(self.num_agents + 1, dtype=np.float32)
        
        # Calculate net energy balance
        net_energy = solar_generation - current_demand
        
        # Rule 1: Handle low battery - buy from grid
        if own_energy < self.low_threshold:
            # Buy from grid (positive = buy)
            deficit = self.low_threshold - own_energy
            action[-1] = min(deficit * 2.0, 1.0)  # Scale to action range
        
        # Rule 2: Handle high battery - sell or share
        elif own_energy > self.high_threshold:
            surplus = own_energy - self.high_threshold
            
            # First, check if any neighbor needs energy
            for i, neighbor_energy in enumerate(neighbor_energies):
                if i == agent_idx:
                    continue  # Skip self
                
                if neighbor_energy < self.critical_threshold:
                    # Share with critically low neighbor (negative = send)
                    share_amount = min(surplus, 0.3)
                    action[i] = -share_amount * 2.0  # Scale to action range
                    surplus -= share_amount
                    
                    if surplus <= 0:
                        break
            
            # If still have surplus and price is good, sell to grid
            if surplus > 0.1 and current_price > 0.5:
                action[-1] = -surplus  # Negative = sell
        
        # Rule 3: Time-based adjustments
        # During peak solar (10am-2pm, time ~0.42-0.58), store more
        if 0.42 <= time_of_day <= 0.58:
            # Prefer storing during peak solar
            if own_energy < 0.7 and net_energy > 0:
                action[-1] = max(action[-1], 0.3)  # Slight grid buy to store
        
        # During evening peak (6pm-9pm, time ~0.75-0.875), be conservative
        elif 0.75 <= time_of_day <= 0.875:
            # Avoid selling during evening peak demand
            if action[-1] < 0:
                action[-1] *= 0.5
        
        # Rule 4: Cooperative behavior - help struggling neighbors
        for i, neighbor_energy in enumerate(neighbor_energies):
            if i == agent_idx:
                continue
            
            # If we have excess and neighbor is low
            if own_energy > 0.5 and neighbor_energy < 0.25:
                share = min((own_energy - 0.5) * 0.5, 0.2)
                action[i] = min(action[i], -share)  # Send energy
        
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action


def get_heuristic_action(
    observations: Dict[str, np.ndarray],
    num_agents: int = 5,
    max_capacity: float = 100.0
) -> Dict[str, np.ndarray]:
    """
    Get heuristic actions for all agents.
    
    Args:
        observations: Dict mapping agent_id to observation array
        num_agents: Number of agents in environment
        max_capacity: Maximum battery capacity
    
    Returns:
        Dict mapping agent_id to action array
    """
    policy = HeuristicPolicy(num_agents=num_agents, max_capacity=max_capacity)
    actions = {}
    
    for agent_id, obs in observations.items():
        # Extract agent index from id (e.g., "agent_0" -> 0)
        agent_idx = int(agent_id.split('_')[1])
        actions[agent_id] = policy.compute_action(obs, agent_idx)
    
    return actions

