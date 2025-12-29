# Architecture Documentation

## System Overview

The Smart Grid MARL system consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  (Next.js + TypeScript + D3.js + TailwindCSS)               │
│                                                              │
│  - Real-time Visualization                                   │
│  - Agent Status Monitoring                                   │
│  - Metrics Dashboard                                         │
└──────────────────┬──────────────────────────────────────────┘
                   │ WebSocket + REST API
┌──────────────────┴──────────────────────────────────────────┐
│                      Backend API                             │
│              (FastAPI + WebSocket)                           │
│                                                              │
│  - Simulation Control                                        │
│  - Real-time State Broadcasting                              │
│  - Model Inference                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│              RL Training & Environment                       │
│      (Ray RLlib + PettingZoo + PyTorch)                     │
│                                                              │
│  - Multi-Agent Environment                                   │
│  - PPO Training Algorithm                                    │
│  - Agent Policies                                            │
└─────────────────────────────────────────────────────────────┘
```

## Backend Architecture

### 1. Environment Layer (`backend/environment/`)

**SmartGridEnv** - Custom PettingZoo Environment

```python
SmartGridEnv(ParallelEnv)
├── State Space
│   ├── Own energy level
│   ├── Demand/generation
│   ├── Grid stability
│   └── Neighbor states
│
├── Action Space
│   ├── Energy transfers (continuous)
│   └── Grid trading (continuous)
│
├── Dynamics
│   ├── Energy balance equations
│   ├── Transfer processing
│   └── Stability calculation
│
└── Rewards
    ├── Grid stability (global)
    ├── Energy efficiency (local)
    ├── Cost optimization (local)
    └── Cooperation bonus
```

**Key Features:**
- Parallel execution for all agents
- Realistic demand/solar patterns
- Multi-objective reward structure
- Configurable grid parameters

### 2. Training Layer (`backend/training/`)

**Ray RLlib Configuration**

```
Training Pipeline
├── Environment Registration
│   └── PettingZoo → RLlib wrapper
│
├── PPO Configuration
│   ├── Policy Networks (one per agent)
│   ├── Training Hyperparameters
│   └── Multi-agent Setup
│
├── Training Loop
│   ├── Rollout Workers (parallel)
│   ├── GPU Acceleration
│   ├── Checkpoint Management
│   └── Metrics Logging
│
└── Outputs
    ├── Trained Models
    ├── Checkpoints
    └── TensorBoard Logs
```

**Multi-Agent Setup:**
- Independent learning approach
- Separate policy network per agent
- Shared environment, local observations
- Cooperation emerges through reward shaping

### 3. API Layer (`backend/api/`)

**FastAPI Server**

```
API Endpoints
├── REST API
│   ├── GET  /              (Status)
│   ├── GET  /api/status    (System info)
│   ├── GET  /api/agents    (Agent info)
│   ├── POST /api/simulation/start
│   └── POST /api/simulation/stop
│
└── WebSocket
    └── /ws/grid           (Real-time updates)
```

**WebSocket Protocol:**

Client → Server:
- Connection requests
- Command messages

Server → Client:
```json
{
  "timestamp": "2024-12-05T...",
  "step": 150,
  "stability": 0.87,
  "agents": [...],
  "energy_flows": [...],
  "metrics": {...}
}
```

**Connection Manager:**
- Handles multiple simultaneous clients
- Broadcasts state updates (10 Hz)
- Automatic reconnection support

## Frontend Architecture

### Component Hierarchy

```
App (page.tsx)
├── ControlPanel
│   ├── Start/Stop buttons
│   └── Connection status
│
├── MetricsDashboard
│   ├── Stability indicator
│   ├── Mean energy
│   ├── Generation totals
│   └── Mean reward
│
├── GridVisualization (D3.js)
│   ├── Agent nodes
│   │   ├── Energy level arcs
│   │   ├── Status colors
│   │   └── Labels
│   ├── Energy flows
│   │   ├── Animated lines
│   │   └── Direction indicators
│   └── Grid stability bar
│
└── AgentCards (Sidebar)
    └── Individual agent status
```

### State Management

```typescript
// Main App State
{
  gridState: GridState | null,
  isConnected: boolean,
  isSimulationRunning: boolean,
  ws: GridWebSocket | null
}

// Grid State (from backend)
{
  timestamp: string,
  step: number,
  stability: number,
  agents: AgentState[],
  energy_flows: EnergyFlow[],
  metrics: GridMetrics
}
```

### WebSocket Client

```
GridWebSocket Class
├── Connection Management
│   ├── Auto-connect
│   ├── Reconnection logic
│   └── State tracking
│
├── Message Handling
│   ├── JSON parsing
│   ├── Type safety
│   └── Error handling
│
└── Event Emitters
    ├── onMessage(handler)
    ├── onError(handler)
    └── onConnect(handler)
```

## Data Flow

### Training Phase

```
1. Environment Creation
   SmartGridEnv initialized with config
   ↓
2. Ray RLlib Setup
   PPO policies created for each agent
   ↓
3. Training Loop (500 iterations)
   For each iteration:
     - Rollout workers collect experiences
     - GPU performs policy updates
     - Metrics logged
     - Best models saved
   ↓
4. Trained Models
   Saved to backend/models/saved_models/
```

### Simulation Phase

```
1. User Clicks "Start"
   Frontend → POST /api/simulation/start
   ↓
2. Backend Initializes
   - Create environment
   - Load trained models (or use random)
   - Start simulation task
   ↓
3. Simulation Loop
   For each step:
     - Get observations
     - Agents select actions
     - Environment steps
     - Calculate rewards
     - Broadcast state via WebSocket
   ↓
4. Frontend Updates
   - Receive grid state via WebSocket
   - Update D3.js visualization
   - Update metrics
   - Update agent cards
```

## Key Design Decisions

### 1. Independent Learning (vs. Centralized Training)

**Choice**: Each agent has its own PPO instance

**Rationale**:
- Simpler implementation
- Better supported by Ray RLlib
- Still allows cooperation through reward shaping
- Scales linearly with agent count

### 2. Continuous Action Space

**Choice**: All actions in [-1, 1] range

**Rationale**:
- PPO handles continuous actions well
- More realistic than discrete choices
- Easier to learn than hybrid spaces

### 3. Real-time WebSocket Updates

**Choice**: 10 Hz broadcast rate

**Rationale**:
- Smooth visualization
- Low enough to avoid network congestion
- Responsive user experience

### 4. D3.js for Visualization

**Choice**: D3.js over canvas or WebGL

**Rationale**:
- Flexible and powerful
- Good integration with React
- Easier to style and customize
- Sufficient performance for 5 agents

## Performance Characteristics

### Training
- **Time**: 4-12 hours for 500 iterations
- **Memory**: ~4GB GPU, ~8GB RAM
- **Convergence**: Typically around iteration 300-400

### Simulation
- **Update Rate**: 10 Hz (100ms per step)
- **Latency**: <50ms WebSocket roundtrip
- **Browser Performance**: 60fps visualization

### Scalability
- **Agents**: Tested with 3-5, could scale to 10-15
- **Clients**: Supports multiple simultaneous viewers
- **Training**: Parallelized across CPU cores

## Security Considerations

**Current Implementation** (Development):
- No authentication
- CORS allows localhost only
- WebSocket unencrypted

**Production Recommendations**:
- Add API authentication
- Use WSS (WebSocket Secure)
- Implement rate limiting
- Add HTTPS
- Restrict CORS origins

## Future Extensions

Potential enhancements:

1. **Algorithm**: Add QMIX for better coordination
2. **Visualization**: 3D grid representation
3. **Features**: 
   - Historical playback
   - Scenario comparison
   - Custom reward weights
   - Live training view
4. **Deployment**: Docker containers, cloud hosting







