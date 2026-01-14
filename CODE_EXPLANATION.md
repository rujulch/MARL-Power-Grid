# Code Explanation - Smart Grid MARL Project

A comprehensive guide to understanding the codebase structure, component interactions, and implementation details.

---

## System Overview

This project implements a multi-agent reinforcement learning system where 5 autonomous neighborhood agents learn to cooperatively manage energy distribution in a smart grid. Each agent has:
- Solar panels (renewable energy generation)
- Battery storage (local energy buffer)
- Variable demand (consumption load)
- Ability to trade energy with neighbors and the grid

The agents learn optimal policies through trial-and-error interaction with the environment, receiving rewards for maintaining grid stability, efficiency, and cost minimization.

---

## Project Structure

```
energy-grid-marl/
├── backend/                    # Python backend (simulation, training, API)
│   ├── environment/            # Custom PettingZoo multi-agent environment
│   │   ├── smart_grid_env.py   # Core environment logic
│   │   └── config.py           # Hyperparameters and settings
│   ├── training/               # RL training scripts
│   │   ├── train_marl.py       # Main training loop with Ray RLlib
│   │   └── train_simple.py     # Simplified training alternative
│   ├── api/                    # FastAPI server
│   │   ├── main.py             # REST + WebSocket endpoints
│   │   └── comparison.py       # Policy comparison experiments
│   ├── policies/               # Baseline policies
│   │   ├── heuristic.py        # Rule-based policy
│   │   └── random_policy.py    # Random action baseline
│   ├── experiments/            # Analysis scripts
│   │   └── scalability.py      # Agent count scaling tests
│   ├── models/                 # Saved model checkpoints
│   └── data/                   # Data loaders and storage
│
├── frontend/                   # Next.js visualization dashboard
│   ├── src/
│   │   ├── app/                # Main application pages
│   │   │   ├── page.tsx        # Dashboard home page
│   │   │   ├── layout.tsx      # App layout wrapper
│   │   │   └── globals.css     # Global styles
│   │   ├── components/         # React UI components
│   │   │   ├── GridVisualization.tsx    # D3.js network graph
│   │   │   ├── AgentCard.tsx            # Individual agent display
│   │   │   ├── MetricsDashboard.tsx     # Performance metrics
│   │   │   ├── ControlPanel.tsx         # Simulation controls
│   │   │   ├── ComparisonDashboard.tsx  # Policy comparison view
│   │   │   ├── DailyComparisonTable.tsx # Multi-day results table
│   │   │   ├── TrainingCurves.tsx       # Training progress charts
│   │   │   ├── ScalabilityChart.tsx     # Scalability analysis
│   │   │   ├── ExportButtons.tsx        # Data export controls
│   │   │   ├── InfoPanel.tsx            # Information display
│   │   │   └── ObjectivesPanel.tsx      # Objective explanations
│   │   ├── lib/                # Utility functions
│   │   │   ├── websocket.ts    # WebSocket client
│   │   │   ├── api.ts          # REST API client
│   │   │   └── utils.ts        # Helper functions
│   │   └── types/              # TypeScript definitions
│   │       └── grid.ts         # Data type interfaces
│   └── package.json            # Frontend dependencies
│
└── docs/                       # Documentation
    ├── ARCHITECTURE.md
    ├── FORMAL_FORMULATION.md
    ├── PROBLEM_EXPLAINED.md
    └── SETUP_GUIDE.md
```

---

## Backend Components

### 1. Environment: `backend/environment/smart_grid_env.py`

**Purpose**: Defines the multi-agent smart grid simulation world.

**Class Structure**:
```python
class SmartGridEnv(ParallelEnv):
    """PettingZoo parallel environment for multi-agent RL."""
    
    def __init__(self, num_agents=5, max_steps=288, ...):
        # Initialize grid parameters, agent list, state tracking
    
    def observation_space(self, agent) -> spaces.Box:
        # Returns observation space: 6 + num_agents dimensions
        # [energy_level, demand, generation, stability, price, time, neighbor_levels...]
    
    def action_space(self, agent) -> spaces.Box:
        # Returns action space: num_agents dimensions
        # [transfer_to_agent_0, transfer_to_agent_1, ..., grid_trade]
    
    def reset(self, seed=None) -> tuple[observations, infos]:
        # Initialize new episode (24-hour day)
        # Generate demand and solar patterns
    
    def step(self, actions) -> tuple[obs, rewards, terms, truncs, infos]:
        # Process one timestep (5 minutes)
        # Apply actions, update states, calculate rewards
    
    def _calculate_reward(self, agent, ...) -> float:
        # Multi-objective reward: stability + efficiency + cost + cooperation
    
    def _calculate_grid_stability(self) -> float:
        # Weighted metric: mean_level + variance + critical_lows + overfull
```

**Neighborhood Differentiation**:
The environment models 5 distinct neighborhood profiles:
- Residential: High evening demand, moderate solar
- Commercial: High daytime demand, limited roof space
- Industrial: Steady demand, large solar installations
- Mixed: Balanced profile
- Suburban: Low base demand, optimal solar orientation

**Reward Components**:
1. Grid Stability (weight: 10.0) - Global cooperation metric
2. Energy Efficiency (-abs(level - 50%)/10) - Maintain optimal storage
3. Cost Minimization - Penalize grid imports, reward selling
4. Cooperation Bonus (0.5 * transfer_amount) - Encourage sharing
5. Shortage Penalty - Heavy penalty for unmet demand

---

### 2. Configuration: `backend/environment/config.py`

**Purpose**: Centralized hyperparameters for easy experimentation.

**Key Configurations**:
```python
ENV_CONFIG = {
    "num_agents": 5,
    "max_steps": 288,           # 24 hours at 5-min intervals
    "max_energy_capacity": 100, # kWh battery capacity
    "max_transfer_rate": 20,    # kWh per timestep
    "base_energy_price": 0.15,  # $/kWh
}

TRAINING_CONFIG = {
    "train_batch_size": 4000,
    "learning_rate": 5e-5,
    "gamma": 0.99,              # Discount factor
    "entropy_coeff": 0.01,      # Exploration bonus
    "num_sgd_iter": 10,         # PPO epochs per update
}
```

---

### 3. Training: `backend/training/train_marl.py`

**Purpose**: Trains agents using PPO with Ray RLlib.

**Training Flow**:
```
1. Initialize Ray (distributed computing framework)
2. Register custom environment with RLlib
3. Configure PPO algorithm:
   - Set multi-agent policies (one per agent or shared)
   - Configure neural network architecture
   - Set training hyperparameters
4. Training loop:
   For each iteration:
     a. Collect experiences (agents interact with environment)
     b. Compute advantages using GAE
     c. Update policies via PPO clipped objective
     d. Log metrics and save checkpoints
5. Save best model based on mean episode reward
```

**Key Classes**:
```python
class TrainingHistory:
    """Tracks and saves training metrics to JSON."""
    
    def add_iteration(self, iteration, result, best_reward):
        # Store metrics: reward, episode_length, policy_rewards
    
    def save(self):
        # Write training_history.json for visualization
```

**Parameter Sharing Option**:
- Default: Each agent has independent neural network
- Optional: All agents share one network (improves sample efficiency)

---

### 4. API Server: `backend/api/main.py`

**Purpose**: FastAPI server connecting frontend to backend.

**Key Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Server and simulation status |
| `/api/agents` | GET | List all agents with properties |
| `/api/simulation/start` | POST | Start multi-day simulation |
| `/api/simulation/stop` | POST | Stop running simulation |
| `/api/comparison/run` | POST | Run policy comparison experiment |
| `/api/training-history` | GET | Retrieve training metrics |
| `/api/scalability` | GET | Get scalability analysis |
| `/api/export/csv` | GET | Download results as CSV |
| `/api/export/json` | GET | Download results as JSON |
| `/ws/grid` | WebSocket | Real-time state streaming |

**Simulation Loop** (in `run_simulation`):
```python
async def run_simulation(config):
    for day in range(num_days):
        env.reset()  # Start new day
        
        for step in range(steps_per_day):
            actions = get_trained_actions(observations, env)
            observations, rewards, ... = env.step(actions)
            
            # Track metrics
            # Broadcast state via WebSocket
            await manager.broadcast(grid_state)
            await asyncio.sleep(0.1)  # 10 Hz update rate
        
        # Send daily summary
```

**Model Inference**:
The server attempts to load trained PPO checkpoints for inference. If unavailable, falls back to random policy.

---

### 5. Policy Comparison: `backend/api/comparison.py`

**Purpose**: Quantitative comparison of trained vs. baseline policies.

**Compared Policies**:
1. **Trained PPO**: Loaded from checkpoint
2. **Heuristic**: Rule-based (share when >60%, request when <40%)
3. **Random**: Uniform random actions

**Metrics Collected**:
- Episode reward (mean, std)
- Grid stability
- Grid imports (kWh)
- Solar utilization (%)
- Demand satisfaction (%)

**Statistical Tests**: Welch's t-test for significance (p-values)

---

### 6. Baseline Policies: `backend/policies/`

**Heuristic Policy** (`heuristic.py`):
```python
def get_action(observation, action_space):
    energy_level = observation[0]
    
    if energy_level > 0.6:
        # Have excess: offer to share
        return negative_transfer_actions
    elif energy_level < 0.4:
        # Need energy: request from neighbors
        return positive_transfer_actions
    else:
        # Balanced: no transfers
        return zero_actions
```

**Random Policy** (`random_policy.py`):
```python
def get_action(action_space):
    return action_space.sample()  # Uniform random
```

---

## Frontend Components

### 7. Main Dashboard: `frontend/src/app/page.tsx`

**Purpose**: Central control interface displaying all visualizations.

**Component Layout**:
```
┌─────────────────────────────────────────────────────────┐
│                    Control Panel                         │
│  [Start] [Stop] [Days: 7]  Connected: ●                 │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │  Grid Visual    │  │  Metrics Panel  │               │
│  │  (D3.js graph)  │  │  Stability: 87% │               │
│  │                 │  │  Energy: 52 kWh │               │
│  └─────────────────┘  └─────────────────┘               │
├─────────────────────────────────────────────────────────┤
│  Agent Cards (5x): Energy bars, demand, generation      │
├─────────────────────────────────────────────────────────┤
│  Daily Comparison Table (after simulation complete)     │
└─────────────────────────────────────────────────────────┘
```

**State Management**:
- WebSocket connection for real-time updates
- Local state for simulation configuration
- Ref-based state for D3.js animation

---

### 8. Grid Visualization: `frontend/src/components/GridVisualization.tsx`

**Purpose**: D3.js network diagram showing agent topology and energy flows.

**Visual Elements**:
- **Nodes**: Circular markers for each agent
  - Color: Green (>60%), Orange (30-60%), Red (<30%) based on energy
  - Size: Fixed, represents neighborhood
- **Edges**: Lines between agents showing transfer relationships
  - Thickness: Proportional to transfer amount
  - Animation: Flow direction indicates sender/receiver
- **Central Grid**: Optional node representing main grid connection

**D3.js Update Cycle**:
```javascript
useEffect(() => {
    // On data change:
    // 1. Update node positions (force simulation)
    // 2. Update node colors based on energy levels
    // 3. Update edge weights based on transfers
    // 4. Animate flow directions
}, [agentData]);
```

---

### 9. Agent Card: `frontend/src/components/AgentCard.tsx`

**Purpose**: Individual agent status display.

**Displayed Information**:
- Agent name (Neighborhood 1-5)
- Energy level (progress bar + percentage)
- Current demand (kWh)
- Solar generation (kWh)
- Step reward
- Cumulative reward

---

### 10. Metrics Dashboard: `frontend/src/components/MetricsDashboard.tsx`

**Purpose**: System-wide performance metrics.

**Displayed Metrics**:
- Grid Stability (%)
- Mean Energy Level (kWh)
- Total Demand (kWh)
- Total Generation (kWh)
- Mean Reward
- Solar Utilization (%)
- Demand Satisfaction (%)

---

### 11. Daily Comparison Table: `frontend/src/components/DailyComparisonTable.tsx`

**Purpose**: Multi-day simulation results in tabular format.

**Table Columns**:
| Day | Reward | Stability | Grid Imports | Solar Util. | Demand Sat. |
|-----|--------|-----------|--------------|-------------|-------------|
| 1   | 2450   | 85%       | 120 kWh      | 72%         | 94%         |
| 2   | 2680   | 88%       | 95 kWh       | 78%         | 96%         |
| ... | ...    | ...       | ...          | ...         | ...         |

---

### 12. Comparison Dashboard: `frontend/src/components/ComparisonDashboard.tsx`

**Purpose**: Policy comparison visualization.

**Displays**:
- Bar charts comparing Trained vs. Heuristic vs. Random
- Metrics: Reward, Stability, Grid Imports
- Statistical significance indicators
- Improvement percentages

---

### 13. Training Curves: `frontend/src/components/TrainingCurves.tsx`

**Purpose**: Visualize training progress over iterations.

**Charts**:
- Episode Reward vs. Iteration (line chart)
- Episode Length vs. Iteration
- Per-policy rewards (if available)

---

### 14. WebSocket Client: `frontend/src/lib/websocket.ts`

**Purpose**: Manages real-time connection to backend.

**Features**:
- Auto-reconnect on disconnect
- Message parsing (JSON to typed objects)
- Event handlers for different message types

**Usage**:
```typescript
const ws = new WebSocket('ws://localhost:8000/ws/grid');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'state_update') {
        updateVisualization(data);
    } else if (data.type === 'day_complete') {
        showDailySummary(data.summary);
    } else if (data.type === 'simulation_complete') {
        showFinalResults(data);
    }
};
```

---

### 15. Type Definitions: `frontend/src/types/grid.ts`

**Purpose**: TypeScript interfaces ensuring type safety.

**Key Types**:
```typescript
interface AgentState {
    id: string;
    name: string;
    energy_level: number;
    demand: number;
    generation: number;
    x: number;
    y: number;
    reward: number;
    cumulative_reward: number;
}

interface GridState {
    timestamp: string;
    step: number;
    current_day: number;
    stability: number;
    agents: AgentState[];
    energy_flows: EnergyFlow[];
    metrics: GridMetrics;
}

interface DailySummary {
    day: number;
    total_reward: number;
    avg_stability: number;
    grid_imports: number;
    solar_utilization: number;
    demand_satisfaction: number;
    agents: AgentDailyMetrics[];
}
```

---

## Data Flow

### Real-Time Simulation:
```
Backend Environment
    ↓ (every 0.1s)
SmartGridEnv.step()
    ↓
API Formats GridState
    ↓ (WebSocket)
Frontend WebSocket Client
    ↓
React State Update
    ↓
Component Re-renders
    ↓
D3.js Updates Visualization
```

### Training:
```
train_marl.py
    ↓
Ray RLlib PPOConfig
    ↓
Create PPO Algorithm
    ↓
Training Loop (500 iterations)
    │
    ├─→ Collect experiences
    ├─→ Compute advantages (GAE)
    ├─→ Update policy (PPO)
    ├─→ Log metrics (TrainingHistory)
    └─→ Save checkpoints
    ↓
Best model → backend/models/saved_models/best/
Training history → backend/models/training_history.json
```

---

## Key Algorithms

### Grid Stability Calculation

```python
def _calculate_grid_stability(self) -> float:
    levels = [agent.energy_level for agent in agents]
    
    # Component 1: Mean energy (target: 50%)
    mean_score = 1.0 - abs(mean(levels) - 0.5) * 2.5
    
    # Component 2: Variance (lower is better)
    variance_score = 1.0 - variance(levels) * 5
    
    # Component 3: Critical lows (<20%)
    critical_score = 1.0 - count(levels < 0.2) * 0.25
    
    # Component 4: Overfull (>90%)
    overfull_score = 1.0 - count(levels > 0.9) * 0.1
    
    # Weighted combination
    stability = (mean_score * 0.25 + variance_score * 0.35 + 
                 critical_score * 0.25 + overfull_score * 0.15)
    
    return clip(stability, 0.0, 1.0)
```

### PPO Policy Update (Simplified)

```python
# For each agent's policy:
for epoch in range(num_epochs):
    # Compute probability ratio
    ratio = new_prob / old_prob
    
    # Clipped surrogate objective
    surrogate = min(
        ratio * advantage,
        clip(ratio, 1-epsilon, 1+epsilon) * advantage
    )
    
    # Policy loss
    policy_loss = -mean(surrogate)
    
    # Value loss
    value_loss = MSE(predicted_value, returns)
    
    # Entropy bonus (encourages exploration)
    entropy_loss = -entropy_coeff * entropy(policy)
    
    # Total loss
    loss = policy_loss + vf_coeff * value_loss + entropy_loss
    
    optimizer.step(loss)
```

---

## Running the System

### Development Mode:
```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Training (optional)
cd backend
python training/train_marl.py --iterations 50
```

### Production Build:
```bash
cd frontend
npm run build
npm start

cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Configuration Reference

### Modifying System Behavior

| Goal | File | Parameter |
|------|------|-----------|
| Change agent count | `config.py` | `ENV_CONFIG["num_agents"]` |
| Adjust battery capacity | `config.py` | `ENV_CONFIG["max_energy_capacity"]` |
| Modify learning rate | `config.py` | `TRAINING_CONFIG["learning_rate"]` |
| Change reward weights | `smart_grid_env.py` | `_calculate_reward()` |
| Adjust stability formula | `smart_grid_env.py` | `_calculate_grid_stability()` |
| Modify demand patterns | `smart_grid_env.py` | `_generate_demand_patterns()` |
| Change visualization colors | `GridVisualization.tsx` | Color scale definitions |
| Adjust simulation speed | `main.py` | `asyncio.sleep(0.1)` |

---

## Summary

This codebase implements a complete multi-agent reinforcement learning system:

1. **Environment**: Custom PettingZoo environment simulating smart grid dynamics with realistic demand/generation patterns

2. **Training**: Ray RLlib PPO training with multi-agent support, checkpoint management, and training history logging

3. **API**: FastAPI server with WebSocket for real-time updates, REST endpoints for control and data export

4. **Frontend**: Next.js dashboard with D3.js visualization, real-time metrics, and multi-day simulation support

5. **Evaluation**: Policy comparison framework with statistical significance testing

The system demonstrates emergent cooperative behavior through independent learning, where agents discover energy-sharing strategies without explicit coordination protocols.
