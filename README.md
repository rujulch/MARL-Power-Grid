# Multi-Agent Reinforcement Learning for Smart Grid Optimization

A sophisticated multi-agent reinforcement learning system where autonomous agents coordinate to optimize energy distribution in a smart grid, balancing local consumption, renewable generation, and grid stability.

## 🎯 Project Overview

This project demonstrates advanced MARL concepts through a realistic smart grid simulation where multiple neighborhood agents learn to:
- Optimize energy distribution autonomously
- Balance renewable energy integration
- Maintain grid stability through cooperation
- Minimize costs through intelligent trading

## 🏗️ Architecture

### Backend
- **Environment**: Custom PettingZoo multi-agent environment
- **Training**: Ray RLlib with PPO (Proximal Policy Optimization)
- **Framework**: PyTorch for neural networks
- **API**: FastAPI with WebSocket for real-time updates

### Frontend
- **Framework**: Next.js 14 with TypeScript
- **Visualization**: D3.js for network graphs, Recharts for metrics
- **Styling**: TailwindCSS with custom design system
- **Real-time**: WebSocket client for live updates

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU with 4GB+ VRAM (recommended)

### Installation

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd frontend
npm install
```

### Training Agents
```bash
cd backend
python training/train_marl.py
```

### Running the Application

#### Start Backend Server
```bash
cd backend
uvicorn api.main:app --reload
```

#### Start Frontend
```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## 📊 Features

- ✅ Multi-agent reinforcement learning with PPO
- ✅ Custom smart grid environment with realistic dynamics
- ✅ Real-time visualization of energy flows
- ✅ Interactive agent decision monitoring
- ✅ Comparative analysis vs baseline strategies
- ✅ Training metrics and convergence tracking

## 🎓 Technical Details

### Environment Specifications
- **Agents**: 3-5 autonomous neighborhood nodes
- **State Space**: Energy levels, demand forecasts, grid status, neighbor states
- **Action Space**: Continuous energy transfers and grid trading
- **Reward Structure**: Multi-objective (stability + efficiency + cost)

### Training Configuration
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Training Time**: 4-12 hours on GTX 1660 Ti
- **Episodes**: ~500 for convergence
- **Expected Performance**: 15-20% improvement in grid stability

## 📈 Results

Performance metrics and trained models will be available after training completion.

## 📝 License

MIT License

## 🙏 Acknowledgments

Built for graduate school applications to demonstrate MARL and full-stack AI engineering skills.

---

**Status**: 🚧 Under Development
**Timeline**: December 2024





