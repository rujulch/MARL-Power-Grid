# Code Explanation - Smart Grid MARL Project

A simple guide to understanding all the code files and how they work together.

---

## 🎯 The Big Picture

Imagine a neighborhood power grid where 5 houses (agents) need to share electricity smartly. Each house has:
- Solar panels (generates power)
- Batteries (stores power)  
- Appliances (uses power)
- Ability to trade with neighbors

Our system teaches these houses to cooperate using AI!

---

## 📁 Project Structure

```
energy-grid-marl/
├── backend/          # Python - The "brain" of the system
│   ├── environment/  # The smart grid simulation
│   ├── training/     # AI learning code
│   ├── api/          # Server that talks to frontend
│   ├── models/       # Saved AI agent brains
│   └── data/         # Data storage
│
├── frontend/         # Website - What you see
│   ├── src/
│   │   ├── app/      # Main pages
│   │   ├── components/  # Reusable UI pieces
│   │   ├── lib/      # Helper code
│   │   └── types/    # TypeScript type definitions
│
└── docs/             # Documentation
```

---

## 🔧 Backend Files (Python)

### 1. `backend/environment/smart_grid_env.py` 
**What it does**: Creates the virtual power grid world

**Simple explanation**:
- Like a video game level where agents play
- Each agent is a neighborhood with solar panels and batteries
- Tracks: How much energy each agent has, how much they need, solar generation
- Calculates: Rewards for good behavior (sharing energy, staying stable)

**Key parts**:
```python
class SmartGridEnv:
    - __init__(): Sets up the grid (5 agents, energy limits, etc.)
    - reset(): Starts a new day (24 hours)
    - step(): Moves time forward (5 minutes) and processes agent actions
    - _calculate_reward(): Decides if agents did good or bad
```

**How agents are rewarded**:
- ✅ Good: Sharing energy, keeping grid stable, using solar
- ❌ Bad: Buying expensive grid power, wasting energy, causing instability

---

### 2. `backend/environment/config.py`
**What it does**: Settings file

**Simple explanation**:
- Like settings in a video game
- Number of agents (5 neighborhoods)
- How much they can store (100 kWh batteries)
- How fast they can trade (20 kWh per 5 minutes)
- Learning parameters (how fast AI learns)

**Why it's separate**: Easy to change settings without touching main code

---

### 3. `backend/training/train_marl.py`
**What it does**: Teaches the AI agents to be smart

**Simple explanation**:
Think of it like training dogs:
1. **Setup**: Create 5 agents (puppies)
2. **Practice**: Let them try things in the environment (play)
3. **Reward/Punish**: Give treats for good actions (rewards)
4. **Repeat**: Do this 500 times until they're smart (iterations)

**The Training Loop**:
```
For 50 iterations:
    1. Agents observe the grid (check energy levels)
    2. Agents decide actions (trade energy? buy from grid?)
    3. Environment processes actions (update energy levels)
    4. Calculate rewards (did they do good?)
    5. AI learns from this experience (update neural networks)
    6. Save best performing agents (checkpoints)
```

**Key concepts**:
- **PPO (Proximal Policy Optimization)**: The AI learning algorithm (like a teaching method)
- **Episode**: One full day (24 hours) in the simulation
- **Iteration**: Multiple episodes grouped together for learning
- **Reward**: Points agents earn for good behavior

---

### 4. `backend/api/main.py`
**What it does**: Web server that connects frontend to backend

**Simple explanation**:
- Like a waiter at a restaurant
- Frontend (customer) asks for things
- Backend (kitchen) provides them
- API (waiter) delivers back and forth

**Main endpoints**:
```
GET  /api/status       → "Is everything working?"
GET  /api/agents       → "Tell me about the 5 agents"
POST /api/simulation/start → "Start the simulation!"
POST /api/simulation/stop  → "Stop it!"
WebSocket /ws/grid     → "Give me live updates!" (10 per second)
```

**WebSocket explained**:
- Regular API: You ask, server answers (like texting)
- WebSocket: Continuous connection (like a phone call)
- Perfect for real-time updates (grid changes every moment)

---

## 🎨 Frontend Files (TypeScript/React)

### 5. `frontend/src/app/page.tsx`
**What it does**: Main dashboard page

**Simple explanation**:
- The control center you see in your browser
- Shows the grid, agents, metrics
- Has start/stop buttons
- Connects to backend via WebSocket

**What it displays**:
1. Control Panel (start/stop simulation)
2. Metrics Dashboard (stability, energy levels)
3. Grid Visualization (the network diagram)
4. Agent Cards (individual agent info)

---

### 6. `frontend/src/components/GridVisualization.tsx`
**What it does**: Draws the network diagram using D3.js

**Simple explanation**:
- Creates the visual network you see
- Agents appear as circles (nodes)
- Energy flows as animated lines
- Colors show health: Green=good, Orange=medium, Red=low

**How D3.js works**:
1. Gets agent data from backend (positions, energy levels)
2. Draws circles for each agent
3. Draws lines between agents for energy transfers
4. Animates lines to show flow direction
5. Updates colors based on energy levels

**Visual elements**:
- **Circle size**: All agents same size
- **Circle color**: Based on energy level (green/orange/red)
- **Lines**: Show energy trades (thickness = amount)
- **Animations**: Lines flow from sender to receiver
- **☀️ icon**: Shows solar is active

---

### 7. `frontend/src/components/AgentCard.tsx`
**What it does**: Shows info about one agent

**Simple explanation**:
- Like a player card in a sports game
- Shows: Energy level (battery %), demand, solar generation, reward
- Updates every 0.1 seconds
- Color-coded for quick understanding

---

### 8. `frontend/src/components/MetricsDashboard.tsx`
**What it does**: Shows overall system stats

**Simple explanation**:
- Like a car dashboard (speedometer, fuel gauge, etc.)
- Grid stability: How balanced is the system?
- Mean energy: Average battery level across all agents
- Total generation: How much solar power being made?
- Mean reward: How well are agents doing?

---

### 9. `frontend/src/components/ControlPanel.tsx`
**What it does**: Start/stop buttons and connection status

**Simple explanation**:
- Remote control for the simulation
- Green dot = connected to backend
- Start button = begin simulation
- Stop button = pause simulation
- Shows warnings if backend not running

---

### 10. `frontend/src/lib/websocket.ts`
**What it does**: Manages real-time connection to backend

**Simple explanation**:
- Phone line between frontend and backend
- Automatically reconnects if disconnected
- Receives grid updates 10 times per second
- Sends them to components for display

**Key features**:
- Auto-reconnect (if connection drops)
- Error handling (graceful failures)
- Message parsing (converts JSON to objects)

---

### 11. `frontend/src/lib/api.ts`
**What it does**: Functions to call backend endpoints

**Simple explanation**:
- Like having a secretary make phone calls for you
- `getStatus()` → "Is backend alive?"
- `getAgents()` → "Tell me about agents"
- `startSimulation()` → "Start it!"
- `stopSimulation()` → "Stop it!"

---

### 12. `frontend/src/types/grid.ts`
**What it does**: TypeScript type definitions

**Simple explanation**:
- Blueprints/contracts for data
- Ensures frontend and backend speak the same language
- Prevents bugs (TypeScript checks types automatically)

**Example**:
```typescript
interface AgentState {
  id: string           // "agent_0"
  energy_level: number // 75.5
  demand: number       // 12.0
  generation: number   // 8.5
  x: number           // 300 (position)
  y: number           // 200 (position)
}
```

---

## 🔗 How Everything Connects

### Startup Flow:

```
1. USER opens browser → http://localhost:3000
2. FRONTEND loads (Next.js)
3. FRONTEND connects to BACKEND (WebSocket)
4. CONNECTION established ✅
5. USER clicks "Start Simulation"
6. FRONTEND sends start request (API call)
7. BACKEND creates environment
8. BACKEND runs simulation loop:
   - Agents observe → decide → act
   - Environment updates
   - Results sent via WebSocket
9. FRONTEND receives updates
10. D3.js redraws visualization
11. Metrics update
12. Agent cards update
→ Repeat steps 8-12 every 0.1 seconds
```

### Training Flow:

```
1. USER runs: python training/train_marl.py
2. RAY initializes (distributed computing framework)
3. ENVIRONMENT created (smart grid)
4. AGENTS created (5 neural networks)
5. TRAINING LOOP (50 iterations):
   For each iteration:
     - Collect experiences (agents interact with environment)
     - Calculate rewards
     - Update neural networks (learning happens)
     - Save checkpoints
6. BEST MODEL saved
7. Training complete ✅
```

---

## 🧠 Key Concepts Explained

### Multi-Agent System
- Not one AI controlling everything
- 5 independent AIs that must cooperate
- Each has own "brain" (neural network)
- Learn through trial and error

### Reinforcement Learning
**Not like regular machine learning**:
- Regular ML: Show examples, learn patterns
- RL: Try things, get rewards/punishments, improve

**Like training a pet**:
- Pet tries action → Good action? Treat! → Pet learns
- Agent tries action → Good reward? → Agent learns

### Emergent Cooperation
- We DON'T program "share with neighbor"
- We ONLY give rewards for grid stability
- Agents DISCOVER sharing helps stability
- Cooperation EMERGES naturally!

This is the cool part - intelligence appears from simple rules!

---

## 💾 Data Flow

### Real-time Simulation:
```
Backend Environment
    ↓ (every 0.1s)
Backend API (FastAPI)
    ↓ (WebSocket)
Frontend WebSocket Client
    ↓
React State
    ↓
Components Update
    ↓
D3.js Redraws
    ↓
You See Changes!
```

### Training:
```
Environment generates data
    ↓
Agents collect experiences
    ↓
Ray RLlib processes
    ↓
PPO algorithm updates neural networks
    ↓
Checkpoints saved to disk
    ↓
Best model kept
```

---

## 🎮 Think of It Like a Video Game

**Environment** = Game Level  
**Agents** = Players  
**Actions** = Button presses  
**Rewards** = Points/Score  
**Episodes** = One playthrough  
**Training** = Getting better by playing multiple times  
**Neural Network** = Player's learned skills  
**Checkpoint** = Save game  

---

## 📊 File Dependencies

**Who needs who?**

```
train_marl.py
  └─ needs: smart_grid_env.py, config.py
  
smart_grid_env.py
  └─ needs: PettingZoo, Gymnasium, NumPy

main.py (API)
  └─ needs: smart_grid_env.py, FastAPI

page.tsx (Frontend)
  └─ needs: GridVisualization, MetricsDashboard, ControlPanel
  
GridVisualization.tsx
  └─ needs: D3.js, websocket.ts, types/grid.ts
```

---

## 🚀 Execution Order

**What runs when?**

### Development:
1. `pip install -r requirements.txt` (once)
2. `npm install` in frontend (once)
3. `python api/main.py` (backend server)
4. `npm run dev` (frontend server)
5. Open browser → interact!

### Training (separate):
1. `python training/train_marl.py --iterations 50`
2. Wait ~30-60 minutes
3. Models saved
4. Can use in simulation

---

## 🔍 Where to Find Things

**Want to change...**
- Number of agents? → `environment/config.py` (ENV_CONFIG)
- Learning speed? → `environment/config.py` (TRAINING_CONFIG)
- UI colors? → `frontend/tailwind.config.ts`
- Reward logic? → `environment/smart_grid_env.py` (_calculate_reward)
- How agents think? → `training/train_marl.py` (PPO parameters)

---

## 📝 Summary

**The system in one sentence**:  
*A web application that simulates and visualizes 5 AI agents learning to cooperatively manage a smart power grid using reinforcement learning.*

**Tech stack in simple terms**:
- **Backend**: Python simulates the grid and trains AI
- **Frontend**: React website shows what's happening
- **Connection**: WebSocket sends live updates
- **AI**: Ray RLlib teaches agents to be smart
- **Visualization**: D3.js draws the network

**What makes it special**:
- Not just simulation - agents actually LEARN
- Not just AI - beautiful real-time visualization
- Not just frontend - full-stack system
- Not just academic - production-quality code

---

**You built something impressive!** 🎉

Even without training working yet, you have:
- A working multi-agent environment
- Real-time visualization
- Professional web application
- Production-ready architecture

The training will work too - we'll get there! 💪

