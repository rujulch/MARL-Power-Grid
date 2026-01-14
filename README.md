# Multi-Agent Reinforcement Learning for Smart Grid Energy Optimization

A multi-agent reinforcement learning (MARL) system for optimizing energy distribution in smart grids. This project demonstrates how autonomous agents can learn cooperative strategies to balance local energy consumption, renewable generation, and grid stability through decentralized decision-making.

## Abstract

This project implements a multi-agent system where independent neighborhood nodes (agents) learn to cooperatively manage energy distribution using Proximal Policy Optimization (PPO). Each agent observes its local state (battery level, demand, solar generation) and decides on energy transfers with neighbors and grid interactions. Through reinforcement learning, agents discover emergent cooperative behaviors that optimize system-wide objectives without explicit coordination protocols.

## Problem Formulation

### Environment Model

The smart grid is modeled as a Partially Observable Markov Game (POMG) with the following components:

**Agents**: N = 5 neighborhood nodes, each representing a distinct consumption profile (residential, commercial, industrial, mixed, suburban).

**State Space** (per agent):
- Local energy storage level (normalized to [0, 1])
- Current electricity demand
- Solar generation capacity
- Grid stability indicator
- Electricity price (time-varying)
- Time of day
- Neighbor energy levels

**Action Space** (continuous):
- Energy transfer intentions to each neighbor in [-1, 1]
- Grid interaction (buy/sell) in [-1, 1]
- Actions are scaled by maximum transfer rate (20 kWh per timestep)

**Reward Structure** (multi-objective):
1. Grid Stability: Weighted combination of energy balance metrics
2. Energy Efficiency: Penalty for deviation from optimal storage (50%)
3. Cost Minimization: Penalize grid imports, reward self-sufficiency
4. Cooperation Bonus: Incentivize energy sharing between agents

### Grid Stability Metric

Grid stability is computed as a weighted combination of:
- Mean energy level deviation from 50% (weight: 0.25)
- Energy variance across agents (weight: 0.35)
- Critical low penalties for agents below 20% (weight: 0.25)
- Overfull penalties for agents above 90% (weight: 0.15)

This formulation ensures that trained policies demonstrably outperform random baselines.

## System Architecture

```
smart-grid-marl/
├── backend/
│   ├── environment/           # PettingZoo multi-agent environment
│   │   ├── smart_grid_env.py  # Core environment implementation
│   │   └── config.py          # Hyperparameters and settings
│   ├── training/
│   │   └── train_marl.py      # Ray RLlib PPO training loop
│   ├── api/
│   │   ├── main.py            # FastAPI server with WebSocket
│   │   └── comparison.py      # Policy comparison experiments
│   ├── policies/
│   │   ├── heuristic.py       # Rule-based baseline policy
│   │   └── random_policy.py   # Random action baseline
│   └── experiments/
│       └── scalability.py     # Agent count scalability analysis
│
├── frontend/                   # Next.js visualization dashboard
│   └── src/
│       ├── app/               # Main application pages
│       └── components/        # React components for visualization
│
└── docs/                      # Technical documentation
    ├── ARCHITECTURE.md        # System design details
    └── FORMAL_FORMULATION.md  # Mathematical formulation
```

## Technical Implementation

### Multi-Agent Environment

The environment extends PettingZoo's `ParallelEnv` interface for simultaneous agent execution:

- **Episode Structure**: 288 timesteps representing a 24-hour period at 5-minute intervals
- **Demand Patterns**: Differentiated profiles per neighborhood type with realistic diurnal variation
- **Solar Generation**: Time-dependent sinusoidal model with cloud variation
- **Energy Transfer**: Bilateral transfers constrained by storage capacity and transfer limits

### Training Algorithm

**Algorithm**: Proximal Policy Optimization (PPO) via Ray RLlib

**Key Configurations**:
- Independent learning with separate neural networks per agent
- Optional parameter sharing for homogeneous agent scenarios
- GPU-accelerated training with configurable batch sizes
- Automatic checkpoint management with best model tracking

**Training Loop**:
1. Agents collect experience trajectories in the environment
2. PPO updates policy networks using clipped surrogate objective
3. Value function trained with MSE loss
4. Entropy bonus encourages exploration

### API and Real-Time Communication

**Backend**: FastAPI server providing:
- REST endpoints for simulation control
- WebSocket for real-time state streaming at 10 Hz
- Policy comparison endpoint (trained vs. heuristic vs. random)
- CSV/JSON export for experimental results

**Frontend**: Next.js dashboard with:
- D3.js network visualization showing agent topology
- Real-time metrics display (stability, energy levels, rewards)
- Multi-day simulation with daily performance summaries
- Interactive policy comparison visualization

## Installation and Usage

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU recommended (4GB+ VRAM)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Training the Model

```bash
cd backend
python training/train_marl.py --iterations 500 --checkpoint-freq 50
```

Training options:
- `--iterations`: Number of training iterations (default: 500)
- `--checkpoint-freq`: Checkpoint save frequency (default: 50)
- `--no-gpu`: Disable GPU acceleration
- `--share-params`: Enable parameter sharing across agents

### Running the Application

Terminal 1 (Backend):
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Access the dashboard at http://localhost:3000

## Experimental Evaluation

### Policy Comparison

The system supports quantitative comparison between:
1. **Trained PPO Policy**: Multi-agent RL with 500+ training iterations
2. **Heuristic Policy**: Rule-based strategy (share when above 60%, request when below 40%)
3. **Random Policy**: Uniformly sampled actions

Metrics evaluated:
- Cumulative episode reward
- Average grid stability
- Grid import volume (kWh)
- Solar utilization percentage
- Demand satisfaction rate

### Scalability Analysis

The framework supports testing with varying agent counts (3, 5, 7, 10+) to analyze:
- Computational complexity scaling
- Cooperation emergence in larger networks
- Training sample efficiency

## Results

Expected performance after training:
- **Grid Stability**: 15-20% improvement over random baseline
- **Training Convergence**: ~300-400 iterations for stable policy
- **Inference Latency**: <50ms per decision step

Detailed results and trained model checkpoints are saved to `backend/models/saved_models/`.

## Technical Stack

| Component | Technology |
|-----------|------------|
| RL Framework | Ray RLlib 2.9+ |
| Environment | PettingZoo 1.24+ |
| Neural Networks | PyTorch 2.1+ |
| Backend API | FastAPI |
| Frontend | Next.js 14, TypeScript |
| Visualization | D3.js, Recharts |
| Styling | TailwindCSS |

## Documentation

- `docs/ARCHITECTURE.md`: Detailed system architecture
- `docs/FORMAL_FORMULATION.md`: Mathematical problem formulation
- `docs/SETUP_GUIDE.md`: Extended installation instructions
- `CODE_EXPLANATION.md`: Code walkthrough for developers

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
2. Terry, J., et al. "PettingZoo: Gym for Multi-Agent Reinforcement Learning." NeurIPS (2021)
3. Liang, E., et al. "RLlib: Abstractions for Distributed Reinforcement Learning." ICML (2018)

## License

MIT License

## Author

Developed as a portfolio project demonstrating multi-agent reinforcement learning and full-stack AI system development.
