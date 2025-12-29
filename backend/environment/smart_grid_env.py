"""
Custom PettingZoo environment for Smart Grid Multi-Agent RL.

Each agent represents a neighborhood/energy node that must:
- Manage local energy storage
- Balance consumption with renewable generation
- Trade energy with neighbors
- Maintain overall grid stability
"""

from pettingzoo import ParallelEnv
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional
import functools


class SmartGridEnv(ParallelEnv):
    """
    Multi-agent smart grid environment for cooperative energy management.
    
    State Space (per agent):
        - Own energy level (normalized 0-1)
        - Current demand (normalized)
        - Solar generation (normalized)
        - Grid stability indicator (0-1)
        - Neighbor energy levels (normalized, one per agent)
        - Current electricity price
        - Time of day (normalized 0-1)
    
    Action Space (per agent):
        - Energy transfers to/from neighbors (continuous, -1 to 1 per neighbor)
        - Grid interaction: buy/sell (continuous, -1 to 1)
    
    Reward Structure:
        - Grid stability (global benefit)
        - Energy efficiency (maintain optimal storage)
        - Cost minimization (avoid expensive grid imports)
        - Cooperation bonus (successful energy trades)
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "smart_grid_v0"
    }
    
    def __init__(
        self,
        num_agents: int = 5,
        max_steps: int = 288,  # 24 hours at 5-minute intervals
        max_energy_capacity: float = 100.0,  # kWh
        max_transfer_rate: float = 20.0,  # kWh per timestep
        base_energy_price: float = 0.15,  # $/kWh
        grid_import_penalty: float = 1.5,
        render_mode: Optional[str] = None,
        use_real_data: bool = False  # Use real energy consumption patterns
    ):
        super().__init__()
        
        # Store as private variable to avoid conflict with PettingZoo's num_agents property
        self._num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.max_steps = max_steps
        self.current_step = 0
        
        # Grid parameters
        self.max_energy_capacity = max_energy_capacity
        self.max_transfer_rate = max_transfer_rate
        self.base_energy_price = base_energy_price
        self.grid_import_penalty = grid_import_penalty
        self.stability_threshold = 0.9
        
        # Rendering
        self.render_mode = render_mode
        
        # Data source
        self.use_real_data = use_real_data
        self._data_loader = None
        if use_real_data:
            try:
                from data.data_loader import EnergyDataLoader
                self._data_loader = EnergyDataLoader()
            except ImportError:
                print("Warning: Could not import data loader. Using synthetic data.")
                self.use_real_data = False
        
        # State tracking
        self.energy_levels = {}
        self.demands = {}
        self.solar_generation = {}
        self.transfer_history = []
        self.stability_history = []
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        """
        Observation space for each agent.
        
        Components:
        - Own energy level (1)
        - Current demand (1)
        - Solar generation (1)
        - Grid stability (1)
        - Current price (1)
        - Time of day (1)
        - Neighbor energy levels (num_agents)
        
        Total: 6 + num_agents dimensions
        """
        obs_dim = 6 + len(self.possible_agents)
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        """
        Action space for each agent.
        
        Components:
        - Energy transfers to each neighbor (num_agents - 1)
        - Grid interaction: buy(+)/sell(-) (1)
        
        Total: num_agents dimensions
        Values in range [-1, 1], scaled by max_transfer_rate
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.possible_agents),),
            dtype=np.float32
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Initialize new episode."""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        # Initialize energy levels (40-60% capacity)
        self.energy_levels = {
            agent: np.random.uniform(40, 60)
            for agent in self.agents
        }
        
        # Generate demand and solar patterns for entire episode
        self.demands = self._generate_demand_patterns()
        self.solar_generation = self._generate_solar_patterns()
        
        # Reset history
        self.transfer_history = []
        self.stability_history = []
        
        # Generate initial observations
        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one timestep of the environment.
        
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        if not actions:
            raise ValueError("Actions dictionary is empty")
        
        # Process energy transfers between agents
        transfers = self._process_energy_transfers(actions)
        
        # Update each agent's energy level
        rewards = {}
        for agent in self.agents:
            # Get current dynamics
            demand = self.demands[agent][self.current_step]
            generation = self.solar_generation[agent][self.current_step]
            
            # Get agent's action
            action = actions[agent]
            grid_action = action[-1]  # Last dimension is grid interaction
            
            # Calculate net energy change
            transfer_net = transfers.get(agent, 0.0)
            grid_trade = grid_action * self.max_transfer_rate
            
            # Update energy level
            energy_change = generation - demand + transfer_net + grid_trade
            self.energy_levels[agent] = np.clip(
                self.energy_levels[agent] + energy_change,
                0.0,
                self.max_energy_capacity
            )
            
            # Calculate reward
            rewards[agent] = self._calculate_reward(
                agent,
                transfer_net,
                grid_trade,
                demand,
                generation
            )
        
        # Record stability
        current_stability = self._calculate_grid_stability()
        self.stability_history.append(current_stability)
        
        # Increment step
        self.current_step += 1
        
        # Generate new observations
        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }
        
        # Check termination
        terminations = {agent: False for agent in self.agents}
        truncations = {
            agent: self.current_step >= self.max_steps
            for agent in self.agents
        }
        
        # Create info dict
        infos = {
            agent: {
                "energy_level": self.energy_levels[agent],
                "grid_stability": current_stability,
                "step": self.current_step,
                "demand": self.demands[agent][min(self.current_step, self.max_steps - 1)],
                "generation": self.solar_generation[agent][min(self.current_step, self.max_steps - 1)]
            }
            for agent in self.agents
        }
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_observation(self, agent: str) -> np.ndarray:
        """Generate observation for specific agent."""
        obs = []
        
        # Own state
        obs.append(self.energy_levels[agent] / self.max_energy_capacity)
        
        # Current dynamics (normalized)
        current_demand = self.demands[agent][min(self.current_step, self.max_steps - 1)]
        current_generation = self.solar_generation[agent][min(self.current_step, self.max_steps - 1)]
        obs.append(np.clip(current_demand / 20.0, 0, 1))
        obs.append(np.clip(current_generation / 15.0, 0, 1))
        
        # Grid state
        obs.append(self._calculate_grid_stability())
        
        # Current price (varies with time)
        obs.append(self._get_current_price())
        
        # Time of day
        obs.append(self.current_step / self.max_steps)
        
        # Neighbor states
        for other_agent in self.agents:
            obs.append(self.energy_levels[other_agent] / self.max_energy_capacity)
        
        return np.array(obs, dtype=np.float32)
    
    def _process_energy_transfers(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Process energy transfers between agents.
        
        Returns dict mapping agent_id -> net energy received
        """
        transfers = {agent: 0.0 for agent in self.agents}
        
        # Process each agent's transfer intentions
        for i, sender in enumerate(self.agents):
            action = actions[sender]
            
            # Each action dimension (except last) represents transfer to that agent index
            for j, receiver in enumerate(self.agents):
                if i == j:
                    continue
                
                # Transfer amount (negative = send, positive = request)
                transfer_intention = action[j] * self.max_transfer_rate
                
                # Only process if sender wants to send (negative value)
                if transfer_intention < 0:
                    amount = min(
                        abs(transfer_intention),
                        self.energy_levels[sender],
                        self.max_energy_capacity - self.energy_levels[receiver]
                    )
                    
                    transfers[sender] -= amount
                    transfers[receiver] += amount
        
        self.transfer_history.append(transfers.copy())
        return transfers
    
    def _calculate_reward(
        self,
        agent: str,
        transfer_net: float,
        grid_trade: float,
        demand: float,
        generation: float
    ) -> float:
        """
        Calculate multi-objective reward for agent.
        
        Components:
        1. Grid stability (global cooperation)
        2. Energy efficiency (maintain good storage level)
        3. Cost minimization (avoid expensive grid imports)
        4. Cooperation bonus (successful energy sharing)
        """
        # 1. Grid stability reward (shared among all agents)
        stability = self._calculate_grid_stability()
        stability_reward = stability * 10.0
        
        # 2. Energy efficiency (maintain 40-60% capacity)
        energy_level = self.energy_levels[agent]
        target_level = 0.5 * self.max_energy_capacity
        efficiency_penalty = -abs(energy_level - target_level) / 10.0
        
        # 3. Cost penalty for grid imports
        if grid_trade > 0:  # Buying from grid
            cost_penalty = -grid_trade * self.base_energy_price * self.grid_import_penalty
        else:  # Selling to grid
            cost_penalty = abs(grid_trade) * self.base_energy_price * 0.8
        
        # 4. Cooperation bonus for energy sharing
        cooperation_bonus = abs(transfer_net) * 0.5
        
        # 5. Penalty for unmet demand or waste
        balance = energy_level + generation - demand
        if balance < 0:  # Energy shortage
            shortage_penalty = balance * 2.0
        else:
            shortage_penalty = 0.0
        
        total_reward = (
            stability_reward +
            efficiency_penalty +
            cost_penalty +
            cooperation_bonus +
            shortage_penalty
        )
        
        return float(total_reward)
    
    def _calculate_grid_stability(self) -> float:
        """
        Calculate overall grid stability metric (0 to 1).
        
        Stability is higher when:
        - Energy levels are balanced across agents (around 50%)
        - All agents have sufficient energy (above 20%)
        - Variance in energy levels is low
        - No agents are overfull (wasting energy)
        
        This formula is designed to differentiate between good and bad policies.
        Random actions should score significantly lower than trained policies.
        """
        if not self.energy_levels:
            return 0.0
        
        levels = np.array(list(self.energy_levels.values()))
        
        # Component 1: Mean energy level (should be around 50%) - 25% weight
        mean_level = np.mean(levels) / self.max_energy_capacity
        mean_score = max(0, 1.0 - abs(mean_level - 0.5) * 2.5)  # More punishing
        
        # Component 2: Variance (lower is better) - LINEAR penalty, can go to 0 - 35% weight
        variance = np.var(levels) / (self.max_energy_capacity ** 2)
        variance_score = max(0, 1.0 - variance * 5)  # Goes to 0 at variance=0.2
        
        # Component 3: Critical lows - HEAVY penalty per agent below 20% - 25% weight
        critical_threshold = 0.2 * self.max_energy_capacity
        num_critical = np.sum(levels < critical_threshold)
        critical_score = max(0, 1.0 - num_critical * 0.25)  # Each critical = -25%
        
        # Component 4: Overfull penalty (waste) - agents above 90% - 15% weight
        overfull_threshold = 0.9 * self.max_energy_capacity
        num_overfull = np.sum(levels > overfull_threshold)
        overfull_score = max(0, 1.0 - num_overfull * 0.1)
        
        # Weighted combination
        stability = (mean_score * 0.25 + variance_score * 0.35 + 
                    critical_score * 0.25 + overfull_score * 0.15)
        
        return float(np.clip(stability, 0.0, 1.0))
    
    def _get_current_price(self) -> float:
        """Get current electricity price (varies by time of day)."""
        # Price is higher during peak hours (morning and evening)
        hour = (self.current_step / self.max_steps) * 24
        
        # Peak in morning (6-9am) and evening (6-9pm)
        morning_peak = np.exp(-((hour - 7.5) ** 2) / 2)
        evening_peak = np.exp(-((hour - 19.5) ** 2) / 2)
        
        price_multiplier = 1.0 + 0.5 * (morning_peak + evening_peak)
        normalized_price = (price_multiplier - 1.0) / 1.0  # Normalize to [0, 1]
        
        return float(np.clip(normalized_price, 0.0, 1.0))
    
    def _generate_demand_patterns(self) -> Dict[str, np.ndarray]:
        """Generate realistic 24-hour demand patterns for each agent."""
        # Try to use real data if available
        if self.use_real_data and self._data_loader is not None:
            try:
                return self._data_loader.generate_realistic_demand(
                    num_agents=len(self.possible_agents),
                    num_steps=self.max_steps,
                    use_real_data=True
                )
            except Exception as e:
                print(f"Warning: Failed to load real data: {e}. Using synthetic.")
        
        # Fall back to synthetic data with NEIGHBORHOOD DIFFERENTIATION
        demands = {}
        time = np.linspace(0, 24, self.max_steps)
        
        # Different neighborhood profiles (residential, commercial, industrial, mixed, suburban)
        neighborhood_profiles = [
            {"name": "residential", "base": 6.0, "amplitude": 4.0, "peak_shift": -6, "evening_mult": 1.5, "morning_mult": 0.8},
            {"name": "commercial", "base": 10.0, "amplitude": 6.0, "peak_shift": -3, "evening_mult": 0.3, "morning_mult": 1.5},
            {"name": "industrial", "base": 12.0, "amplitude": 3.0, "peak_shift": -4, "evening_mult": 0.2, "morning_mult": 1.2},
            {"name": "mixed", "base": 8.0, "amplitude": 5.0, "peak_shift": -5, "evening_mult": 1.0, "morning_mult": 1.0},
            {"name": "suburban", "base": 5.0, "amplitude": 4.5, "peak_shift": -7, "evening_mult": 1.8, "morning_mult": 0.6},
        ]
        
        for i, agent in enumerate(self.possible_agents):
            # Get profile for this neighborhood (cycle if more agents than profiles)
            profile = neighborhood_profiles[i % len(neighborhood_profiles)]
            
            # Base sinusoidal pattern with neighborhood-specific parameters
            base = profile["base"] + profile["amplitude"] * np.sin(2 * np.pi * (time + profile["peak_shift"]) / 24)
            
            # Add secondary peak for evening (stronger for residential)
            evening_peak = 3.0 * profile["evening_mult"] * np.exp(-((time - 19) ** 2) / 8)
            
            # Add morning peak (stronger for commercial/industrial)
            morning_peak = 2.0 * profile["morning_mult"] * np.exp(-((time - 8) ** 2) / 4)
            
            # Add realistic noise (scaled to base demand)
            noise = np.random.normal(0, 0.8 + profile["base"] * 0.05, self.max_steps)
            
            # Combine and ensure non-negative
            pattern = base + evening_peak + morning_peak + noise
            demands[agent] = np.maximum(pattern, 2.0)
        
        return demands
    
    def _generate_solar_patterns(self) -> Dict[str, np.ndarray]:
        """Generate realistic solar generation patterns."""
        # Try to use realistic patterns from data loader
        if self.use_real_data and self._data_loader is not None:
            try:
                return self._data_loader.generate_realistic_solar(
                    num_agents=len(self.possible_agents),
                    num_steps=self.max_steps
                )
            except Exception as e:
                print(f"Warning: Failed to generate solar from data loader: {e}")
        
        # Fall back to synthetic data with NEIGHBORHOOD DIFFERENTIATION
        solar = {}
        time = np.linspace(0, 24, self.max_steps)
        
        # Different solar capacities per neighborhood (based on roof space, orientation, etc.)
        # Residential/suburban have more roof space, industrial has larger installations
        solar_capacities = [
            {"capacity": 10.0, "efficiency": 0.9},   # residential - medium panels, good orientation
            {"capacity": 8.0, "efficiency": 0.75},   # commercial - less roof space, partial shading
            {"capacity": 15.0, "efficiency": 0.85},  # industrial - large installation
            {"capacity": 11.0, "efficiency": 0.8},   # mixed - varied
            {"capacity": 13.0, "efficiency": 0.95},  # suburban - large roofs, optimal orientation
        ]
        
        for i, agent in enumerate(self.possible_agents):
            # Get solar profile for this neighborhood
            solar_profile = solar_capacities[i % len(solar_capacities)]
            
            # Solar generation: 6am to 6pm, peak at noon
            generation = np.zeros(self.max_steps)
            
            for j, t in enumerate(time):
                if 6 <= t <= 18:
                    # Sinusoidal generation during daylight with neighborhood-specific capacity
                    normalized_time = (t - 6) / 12  # 0 to 1
                    generation[j] = solar_profile["capacity"] * solar_profile["efficiency"] * np.sin(np.pi * normalized_time)
            
            # Add cloud variation (random scaling) - varies per neighborhood
            base_cloud = np.random.uniform(0.6, 1.0)  # Base weather for the day
            cloud_variation = np.random.normal(0, 0.1, self.max_steps)
            cloud_factor = np.clip(base_cloud + cloud_variation, 0.4, 1.0)
            solar[agent] = generation * cloud_factor
        
        return solar
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            print(f"Grid Stability: {self._calculate_grid_stability():.2f}")
            for agent in self.agents:
                print(f"{agent}: Energy={self.energy_levels[agent]:.1f} kWh")
    
    def close(self):
        """Clean up resources."""
        pass





