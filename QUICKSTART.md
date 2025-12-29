# Quick Start Guide

Get the Smart Grid MARL system up and running in 5 minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+
- 16GB RAM
- GPU recommended (but not required)

## Installation

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend
npm install
```

## Running the System

### Terminal 1: Start Backend

```bash
cd backend
python api/main.py
```

Wait for:
```
Smart Grid MARL API Starting...
WebSocket endpoint: ws://localhost:8000/ws/grid
```

### Terminal 2: Start Frontend

```bash
cd frontend
npm run dev
```

Wait for:
```
Local: http://localhost:3000
```

### Terminal 3 (Optional): Train Agents

```bash
cd backend
python training/train_marl.py --iterations 50
```

## Using the Dashboard

1. Open `http://localhost:3000` in your browser
2. Wait for "Connected" status
3. Click "Start Simulation"
4. Watch agents cooperate to optimize the grid!

## What You'll See

- **Network Visualization**: 5 agent nodes with energy levels
- **Energy Flows**: Animated lines showing energy transfers
- **Metrics**: Real-time stability, generation, and demand
- **Agent Cards**: Individual agent status

## Understanding the Visualization

### Agent Nodes
- **Green**: Good energy level (>70%)
- **Orange**: Medium energy level (40-70%)
- **Red**: Low energy level (<40%)
- **☀️**: Solar generation active
- **Red dot**: High demand

### Energy Flows
- **Blue lines**: Energy transfers
- **Animated**: Direction of flow
- **Thickness**: Amount transferred

### Metrics
- **Grid Stability**: How balanced the system is
- **Mean Energy**: Average storage across agents
- **Total Generation**: Solar power produced
- **Mean Reward**: Agent learning performance

## Training Your Own Agents

Quick test (10 minutes):
```bash
python training/train_marl.py --iterations 50
```

Full training (4-12 hours):
```bash
python training/train_marl.py --iterations 500
```

Trained models are saved to:
```
backend/models/saved_models/best_model
```

## Troubleshooting

### Backend won't start
- Check Python version: `python --version` (need 3.10+)
- Activate virtual environment
- Reinstall: `pip install -r requirements.txt`

### Frontend won't start
- Check Node version: `node --version` (need 18+)
- Delete and reinstall: `rm -rf node_modules && npm install`

### "Disconnected" status
- Make sure backend is running on port 8000
- Check for error messages in backend terminal
- Try restarting both backend and frontend

### Simulation won't start
- Check backend terminal for errors
- Make sure WebSocket shows "Connected"
- Try stopping and starting again

## Next Steps

- ✅ Explore different scenarios
- ✅ Train your own agents
- ✅ Modify environment parameters
- ✅ Check out full documentation in `docs/`

## Need Help?

- **Setup Issues**: See `docs/SETUP_GUIDE.md`
- **Architecture**: See `docs/ARCHITECTURE.md`
- **API Reference**: Visit `http://localhost:8000/docs`

---

**Enjoy optimizing the smart grid!** ⚡







