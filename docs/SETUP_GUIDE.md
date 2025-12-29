# Setup Guide - Smart Grid MARL

Complete installation and setup guide for the Multi-Agent Reinforcement Learning Smart Grid system.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
  - Tested on: GTX 1660 Ti (6GB)
  - CPU-only training is possible but slower
- **RAM**: 16GB recommended
- **Storage**: 5GB free space

### Software
- **Python**: 3.10 or higher
- **Node.js**: 18 or higher
- **CUDA**: 11.8+ (if using GPU)
- **Git**: For cloning the repository

## Installation Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd energy-grid-marl
```

### 2. Backend Setup

#### Create Python Virtual Environment

```bash
cd backend
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Ray RLlib 2.9.0 (MARL framework)
- PettingZoo 1.24.3 (multi-agent environments)
- PyTorch 2.1.0 (neural networks)
- FastAPI (web API)
- And other dependencies

**Note**: Installation may take 10-15 minutes due to large packages.

#### Verify Installation

```bash
python -c "import ray; import torch; print('Backend ready!')"
```

### 3. Frontend Setup

#### Install Node Dependencies

```bash
cd ../frontend
npm install
```

This will install:
- Next.js 14
- React 18
- D3.js (visualization)
- TailwindCSS (styling)
- And other dependencies

**Note**: Installation may take 5-10 minutes.

#### Verify Installation

```bash
npm run build
```

Should complete without errors.

## Running the System

### Step 1: Start Backend API

```bash
cd backend
python api/main.py
```

You should see:
```
Smart Grid MARL API Starting...
Environment: 5 agents
API documentation: http://localhost:8000/docs
WebSocket endpoint: ws://localhost:8000/ws/grid
```

The API will be available at `http://localhost:8000`

### Step 2: Start Frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

You should see:
```
- Local:        http://localhost:3000
- ready started server on 0.0.0.0:3000
```

### Step 3: Access Dashboard

Open your browser and navigate to:
```
http://localhost:3000
```

You should see the Smart Grid MARL dashboard.

## Training Agents

### Quick Training (Testing)

For testing purposes, run a short training session:

```bash
cd backend
python training/train_marl.py --iterations 50
```

This will train for 50 iterations (~5-10 minutes on GPU).

### Full Training (Recommended)

For production results:

```bash
python training/train_marl.py --iterations 500
```

This will take 4-12 hours depending on your GPU.

### Training Options

```bash
# Custom number of iterations
python training/train_marl.py --iterations 300

# CPU-only training
python training/train_marl.py --no-gpu

# Custom checkpoint frequency
python training/train_marl.py --checkpoint-freq 25

# Custom results directory
python training/train_marl.py --results-dir my_training_run
```

### Monitor Training

Training progress is displayed in the terminal:

```
=========================================================
Iteration 100/500
=========================================================
Mean Reward:        45.32
Mean Episode Length: 250.5
Episodes Total:     1500
Timesteps Total:    375750

✓ New best model! Reward: 45.32
  Saved to: backend/models/saved_models/best_model
```

### Training Outputs

- **Checkpoints**: Saved in `backend/results/` or custom directory
- **Best Model**: Automatically saved to `backend/models/saved_models/best_model`
- **TensorBoard Logs**: Can be viewed with `tensorboard --logdir results/`

## Troubleshooting

### Backend Issues

#### Ray Init Error
```
Error: Ray failed to initialize
```

**Solution**: Make sure no other Ray processes are running:
```bash
ray stop
```

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size in `backend/environment/config.py`:
```python
TRAINING_CONFIG = {
    "train_batch_size": 2000,  # Reduced from 4000
    "sgd_minibatch_size": 64,   # Reduced from 128
}
```

#### Import Errors
```
ModuleNotFoundError: No module named 'ray'
```

**Solution**: Make sure virtual environment is activated:
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Frontend Issues

#### Port Already in Use
```
Error: Port 3000 is already in use
```

**Solution**: Kill the process or use a different port:
```bash
npm run dev -- -p 3001
```

#### WebSocket Connection Failed
```
WebSocket error: Connection refused
```

**Solution**: 
1. Make sure backend is running on port 8000
2. Check `src/lib/websocket.ts` and `src/lib/api.ts` for correct URLs

#### Module Not Found
```
Cannot find module '@/components/...'
```

**Solution**: 
```bash
rm -rf node_modules .next
npm install
npm run dev
```

## Verification Checklist

Use this checklist to verify everything is working:

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] Backend dependencies installed without errors
- [ ] Backend API starts successfully (port 8000)
- [ ] Node.js 18+ installed
- [ ] Frontend dependencies installed without errors
- [ ] Frontend builds successfully
- [ ] Frontend runs on port 3000
- [ ] Dashboard loads in browser
- [ ] WebSocket shows "Connected" status
- [ ] Can start simulation successfully
- [ ] Grid visualization displays agents
- [ ] Metrics update in real-time

## Next Steps

After successful setup:

1. **Test the System**: Start a simulation to verify everything works
2. **Train Agents**: Run the training script (start with 50 iterations)
3. **Explore API**: Visit `http://localhost:8000/docs` for API documentation
4. **Customize**: Modify parameters in `backend/environment/config.py`

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review terminal outputs for error messages
3. Verify all dependencies are installed correctly
4. Check that ports 3000 and 8000 are available







