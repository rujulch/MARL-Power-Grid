"""
FastAPI server for Smart Grid MARL system.

Provides REST API and WebSocket endpoints for:
- Running trained agent simulations
- Real-time grid state updates
- Performance metrics
- Model information
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import numpy as np
import json
import csv
import io
from datetime import datetime
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from environment.smart_grid_env import SmartGridEnv
from environment.config import ENV_CONFIG
from api.comparison import run_comparison, ComparisonResults
from experiments.scalability import run_scalability_analysis, run_single_scale_test
from dataclasses import asdict

# Try to import Ray for trained model inference
try:
    import ray
    import torch
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Using random policy.")

# Track if trained model is actually working for inference
using_trained_model_actual = False

# Trained model path - check multiple possible locations
def find_trained_model():
    """Find the trained model in various possible locations."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    possible_paths = [
        os.path.join(base_dir, "backend", "models", "saved_models", "best"),  # From train_simple.py
        os.path.join(base_dir, "models", "saved_models", "best"),
        os.path.join(base_dir, "results", "final"),
        os.path.join(base_dir, "backend", "models", "saved_models", "best_model"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check for checkpoint file
            if os.path.exists(os.path.join(path, "rllib_checkpoint.json")):
                return path
    
    return possible_paths[0]  # Return first path even if not found

TRAINED_MODEL_PATH = find_trained_model()

# Global trained algorithm
trained_algo = None

app = FastAPI(
    title="Smart Grid MARL API",
    description="Multi-Agent Reinforcement Learning for Smart Grid Optimization",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class SimulationConfig(BaseModel):
    """Configuration for simulation run."""
    num_days: Optional[int] = 7  # Number of days to simulate
    steps_per_day: Optional[int] = 288  # 288 steps = 24 hours (5-min intervals)
    scenario: Optional[str] = "default"  # default, high_demand, high_solar


class AgentState(BaseModel):
    """State of a single agent."""
    id: str
    energy_level: float
    demand: float
    generation: float
    x: float
    y: float


class GridState(BaseModel):
    """Complete grid state."""
    timestamp: str
    step: int
    stability: float
    agents: List[AgentState]
    energy_flows: List[Dict]


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# Global simulation state
simulation_running = False
current_env = None


def load_trained_model():
    """Load the trained PPO model for inference."""
    global trained_algo
    
    if not RAY_AVAILABLE:
        print("Ray not available, cannot load trained model")
        return False
    
    # Check if model exists
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Trained model not found at {TRAINED_MODEL_PATH}")
        print("Running with random policy instead.")
        return False
    
    try:
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        
        # Register environment
        def env_creator(config):
            return ParallelPettingZooEnv(SmartGridEnv(**config))
        
        register_env("smart_grid", env_creator)
        
        # Get spaces
        temp_env = SmartGridEnv(**ENV_CONFIG)
        obs_space = temp_env.observation_space("agent_0")
        act_space = temp_env.action_space("agent_0")
        num_agents = len(temp_env.possible_agents)
        
        # Build config matching training
        config = PPOConfig()
        config = config.environment("smart_grid", env_config=ENV_CONFIG)
        config = config.framework("torch")
        config = config.resources(num_gpus=0)  # CPU for inference
        
        policies = {f"agent_{i}": (None, obs_space, act_space, {}) for i in range(num_agents)}
        config = config.multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id
        )
        
        # Build and restore
        trained_algo = config.build()
        trained_algo.restore(TRAINED_MODEL_PATH)
        
        print(f"Successfully loaded trained model from {TRAINED_MODEL_PATH}")
        return True
        
    except Exception as e:
        print(f"Error loading trained model: {e}")
        print("Running with random policy instead.")
        return False


def get_trained_actions(observations: Dict, env: SmartGridEnv) -> tuple:
    """Get actions from trained model or random policy.
    
    Returns:
        Tuple of (actions dict, whether trained model was used)
    """
    global trained_algo, using_trained_model_actual
    
    if trained_algo is not None:
        try:
            # New Ray API: Use RLModule directly
            actions = {}
            for agent_id, obs in observations.items():
                # Get the module for this policy
                module = trained_algo.get_module(agent_id)
                # Reshape observation for batch processing
                obs_batch = {"obs": torch.tensor(obs).unsqueeze(0).float()}
                # Forward inference
                with torch.no_grad():
                    output = module.forward_inference(obs_batch)
                    # Debug: print output keys on first call
                    if not hasattr(get_trained_actions, '_logged_output'):
                        print(f"DEBUG: forward_inference output keys: {output.keys()}")
                        for k, v in output.items():
                            print(f"DEBUG: {k} shape: {v.shape if hasattr(v, 'shape') else type(v)}")
                        get_trained_actions._logged_output = True
                    
                    # Try different possible output keys
                    if "action_dist_inputs" in output:
                        dist_inputs = output["action_dist_inputs"].squeeze(0)
                        # For continuous PPO: first half = means, second half = log_stds
                        action_dim = len(dist_inputs) // 2
                        action = dist_inputs[:action_dim].numpy()
                    else:
                        raise ValueError(f"No action_dist_inputs found in {output.keys()}")
                actions[agent_id] = action
            using_trained_model_actual = True
            return actions, True
        except Exception as e:
            import traceback
            print(f"Error getting trained actions: {e}")
            traceback.print_exc()
            using_trained_model_actual = False
    
    # Fallback to random policy
    return {agent: env.action_space(agent).sample() for agent in env.agents}, False


def step_to_time(step: int) -> Dict:
    """Convert simulation step to time of day.
    
    288 steps = 24 hours (5-minute intervals)
    """
    total_minutes = step * 5
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    day = (total_minutes // (24 * 60)) + 1
    
    # Format time
    am_pm = "AM" if hours < 12 else "PM"
    display_hour = hours if hours <= 12 else hours - 12
    if display_hour == 0:
        display_hour = 12
    
    time_str = f"{display_hour}:{minutes:02d} {am_pm}"
    
    return {
        "day": day,
        "hour": hours,
        "minute": minutes,
        "time_string": time_str,
        "period": "Morning" if 5 <= hours < 12 else "Afternoon" if 12 <= hours < 17 else "Evening" if 17 <= hours < 21 else "Night"
    }


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Smart Grid MARL API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "websocket": "/ws/grid",
            "simulation": "/api/simulation",
            "agents": "/api/agents"
        }
    }


@app.get("/api/status")
async def get_status():
    """Get API and simulation status."""
    return {
        "api_status": "running",
        "simulation_running": simulation_running,
        "connected_clients": len(manager.active_connections),
        "environment_config": ENV_CONFIG
    }


@app.get("/api/agents")
async def get_agents():
    """Get information about all agents."""
    agents_info = []
    for i in range(ENV_CONFIG["num_agents"]):
        agents_info.append({
            "id": f"agent_{i}",
            "name": f"Neighborhood {i+1}",
            "max_capacity": ENV_CONFIG["max_energy_capacity"],
            "max_transfer": ENV_CONFIG["max_transfer_rate"]
        })
    return {"agents": agents_info}


@app.post("/api/simulation/start")
async def start_simulation(config: SimulationConfig):
    """Start a new simulation."""
    global simulation_running
    
    if simulation_running:
        raise HTTPException(status_code=400, detail="Simulation already running")
    
    simulation_running = True
    
    # Start simulation in background
    asyncio.create_task(run_simulation(config))
    
    return {
        "status": "started",
        "config": config.dict(),
        "message": "Simulation started. Connect to WebSocket for real-time updates."
    }


@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop running simulation."""
    global simulation_running
    
    simulation_running = False
    
    return {"status": "stopped", "message": "Simulation stopped"}


async def run_simulation(config: SimulationConfig):
    """
    Run multi-day simulation with trained or random policy.
    Resets environment at the start of each day.
    Broadcasts state updates and daily summaries via WebSocket.
    """
    global simulation_running, current_env
    
    try:
        # Create environment
        env = SmartGridEnv(**ENV_CONFIG)
        current_env = env
        
        # Generate agent positions for visualization
        num_agents = ENV_CONFIG["num_agents"]
        agent_positions = _generate_agent_positions(num_agents)
        
        # Check if using trained model
        using_trained = trained_algo is not None
        
        # Track all daily summaries for final comparison
        all_daily_summaries = []
        
        # Total steps across all days
        total_steps = config.num_days * config.steps_per_day
        global_step = 0
        
        # Multi-day simulation loop
        for day in range(1, config.num_days + 1):
            if not simulation_running:
                break
            
            # Reset environment at start of each day
            observations, infos = env.reset()
            
            # Daily metrics tracking
            daily_grid_imports = 0.0
            daily_solar_used = 0.0
            daily_solar_generated = 0.0
            daily_rewards = {agent: 0.0 for agent in env.agents}
            daily_demands_met = 0
            daily_total_demands = 0
            daily_stability_sum = 0.0
            daily_energy_sum = {agent: 0.0 for agent in env.agents}
            
            # Per-agent comprehensive metrics
            agent_metrics = {
                agent: {
                    "total_demand": 0.0,
                    "solar_generated": 0.0,
                    "grid_imported": 0.0,
                    "energy_traded_in": 0.0,
                    "energy_traded_out": 0.0,
                    "battery_start": float(env.energy_levels[agent]),
                    "battery_end": 0.0,
                    "demands_met": 0,
                    "total_demand_events": 0
                }
                for agent in env.agents
            }
            
            # Day simulation loop
            for step in range(config.steps_per_day):
                if not simulation_running:
                    break
                
                # Get actions from trained model or random
                actions, using_trained = get_trained_actions(observations, env)
                
                # Step environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Track daily metrics
                current_stability = env._calculate_grid_stability()
                daily_stability_sum += current_stability
                
                for agent in env.agents:
                    daily_rewards[agent] += rewards.get(agent, 0)
                    daily_energy_sum[agent] += env.energy_levels[agent]
                    generation = infos[agent]["generation"]
                    demand = infos[agent]["demand"]
                    daily_solar_generated += generation
                    daily_solar_used += min(generation, demand)
                    daily_total_demands += 1
                    # Check if agent can meet demand with current energy + generation
                    if env.energy_levels[agent] + generation >= demand:
                        daily_demands_met += 1
                    
                    # Track per-agent metrics
                    agent_metrics[agent]["total_demand"] += demand
                    agent_metrics[agent]["solar_generated"] += generation
                    agent_metrics[agent]["total_demand_events"] += 1
                    # Check if agent can meet demand with current energy + generation
                    if env.energy_levels[agent] + generation >= demand:
                        agent_metrics[agent]["demands_met"] += 1
                    
                    # Track grid imports per agent
                    net = generation - demand
                    if net < 0:
                        agent_grid_import = abs(net) * 0.5
                        agent_metrics[agent]["grid_imported"] += agent_grid_import
                        daily_grid_imports += agent_grid_import
                
                # Track energy trades from actions
                for agent in env.agents:
                    action = actions.get(agent, [0, 0])
                    if len(action) >= 2:
                        trade_amount = action[1] if isinstance(action, (list, np.ndarray)) else 0
                        if trade_amount > 0:
                            agent_metrics[agent]["energy_traded_out"] += abs(trade_amount) * 5
                        else:
                            agent_metrics[agent]["energy_traded_in"] += abs(trade_amount) * 5
                
                # Get simulation time (within current day)
                sim_time = step_to_time(step)
                sim_time["day"] = day
                
                # Calculate overall progress
                global_step = (day - 1) * config.steps_per_day + step
                overall_progress = round((global_step + 1) / total_steps * 100, 1)
                
                # Prepare grid state for frontend
                grid_state = {
                    "type": "state_update",
                    "timestamp": datetime.now().isoformat(),
                    "step": step,
                    "steps_per_day": config.steps_per_day,
                    "current_day": day,
                    "total_days": config.num_days,
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "progress": overall_progress,
                    "day_progress": round((step + 1) / config.steps_per_day * 100, 1),
                    "stability": float(current_stability),
                    "using_trained_model": using_trained,
                    "simulation_time": sim_time,
                    "agents": [
                        {
                            "id": agent,
                            "name": f"Neighborhood {i+1}",
                            "energy_level": float(env.energy_levels[agent]),
                            "demand": float(infos[agent]["demand"]),
                            "generation": float(infos[agent]["generation"]),
                            "x": agent_positions[agent]["x"],
                            "y": agent_positions[agent]["y"],
                            "reward": float(rewards.get(agent, 0)),
                            "cumulative_reward": float(daily_rewards[agent])
                        }
                        for i, agent in enumerate(env.agents)
                    ],
                    "energy_flows": _calculate_energy_flows(env, actions),
                    "metrics": {
                        "mean_energy": float(np.mean([env.energy_levels[a] for a in env.agents])),
                        "total_demand": float(sum([infos[a]["demand"] for a in env.agents])),
                        "total_generation": float(sum([infos[a]["generation"] for a in env.agents])),
                        "mean_reward": float(np.mean(list(rewards.values()))),
                        "cumulative_reward": float(sum(daily_rewards.values())),
                        "grid_imports": float(daily_grid_imports),
                        "solar_utilization": float(daily_solar_used / max(daily_solar_generated, 1) * 100),
                        "demand_satisfaction": float(daily_demands_met / max(daily_total_demands, 1) * 100)
                    }
                }
                
                # Broadcast to all connected clients
                await manager.broadcast(grid_state)
                
                # Control simulation speed (10 updates per second)
                await asyncio.sleep(0.1)
                
                # Check if episode is done early
                if all(truncations.values()) or all(terminations.values()):
                    break
            
            # Update battery end levels
            for agent in env.agents:
                agent_metrics[agent]["battery_end"] = float(env.energy_levels[agent])
            
            # Create daily summary with comprehensive metrics
            daily_summary = {
                "day": day,
                "total_reward": float(sum(daily_rewards.values())),
                "avg_stability": float(daily_stability_sum / config.steps_per_day),
                "grid_imports": float(daily_grid_imports),
                "total_solar_generated": float(daily_solar_generated),
                "total_solar_used": float(daily_solar_used),
                "solar_utilization": float(daily_solar_used / max(daily_solar_generated, 1) * 100),
                "demand_satisfaction": float(daily_demands_met / max(daily_total_demands, 1) * 100),
                "total_demand": float(sum(agent_metrics[a]["total_demand"] for a in env.agents)),
                "agents": [
                    {
                        "agent_id": agent,
                        "name": f"Neighborhood {i+1}",
                        "reward": float(daily_rewards[agent]),
                        "avg_battery_level": float(daily_energy_sum[agent] / config.steps_per_day),
                        "total_demand": float(agent_metrics[agent]["total_demand"]),
                        "solar_generated": float(agent_metrics[agent]["solar_generated"]),
                        "grid_imported": float(agent_metrics[agent]["grid_imported"]),
                        "energy_traded_in": float(agent_metrics[agent]["energy_traded_in"]),
                        "energy_traded_out": float(agent_metrics[agent]["energy_traded_out"]),
                        "battery_start": float(agent_metrics[agent]["battery_start"]),
                        "battery_end": float(agent_metrics[agent]["battery_end"]),
                        "demand_satisfaction": float(
                            agent_metrics[agent]["demands_met"] / 
                            max(agent_metrics[agent]["total_demand_events"], 1) * 100
                        )
                    }
                    for i, agent in enumerate(env.agents)
                ]
            }
            all_daily_summaries.append(daily_summary)
            
            # Broadcast day complete
            await manager.broadcast({
                "type": "day_complete",
                "day": day,
                "total_days": config.num_days,
                "summary": daily_summary,
                "message": f"Day {day} of {config.num_days} completed"
            })
            
            # Small delay between days
            await asyncio.sleep(0.5)
        
        # Simulation complete - send all daily data for comparison
        simulation_running = False
        await manager.broadcast({
            "type": "simulation_complete",
            "message": f"Simulation finished - {config.num_days} days completed",
            "total_days": config.num_days,
            "daily_summaries": all_daily_summaries,
            "overall_metrics": {
                "avg_daily_reward": float(np.mean([s["total_reward"] for s in all_daily_summaries])),
                "avg_stability": float(np.mean([s["avg_stability"] for s in all_daily_summaries])),
                "avg_grid_imports": float(np.mean([s["grid_imports"] for s in all_daily_summaries])),
                "avg_solar_utilization": float(np.mean([s["solar_utilization"] for s in all_daily_summaries])),
                "avg_demand_satisfaction": float(np.mean([s["demand_satisfaction"] for s in all_daily_summaries]))
            }
        })
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        simulation_running = False
        await manager.broadcast({
            "type": "error",
            "message": str(e)
        })


def _generate_agent_positions(num_agents: int) -> Dict:
    """Generate positions for agents in circular layout."""
    positions = {}
    radius = 300
    center_x, center_y = 500, 300
    
    for i in range(num_agents):
        angle = (2 * np.pi * i) / num_agents
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        positions[f"agent_{i}"] = {"x": float(x), "y": float(y)}
    
    return positions


def _calculate_energy_flows(env: SmartGridEnv, actions: Dict) -> List[Dict]:
    """Calculate energy flow connections between agents."""
    flows = []
    
    for i, sender in enumerate(env.agents):
        action = actions[sender]
        
        for j, receiver in enumerate(env.agents):
            if i == j:
                continue
            
            # Transfer amount from action
            transfer = float(action[j])
            
            # Only show significant transfers
            if abs(transfer) > 0.1:
                flows.append({
                    "source": sender,
                    "target": receiver,
                    "amount": transfer,
                    "type": "transfer" if transfer < 0 else "request"
                })
    
    return flows


class ComparisonConfig(BaseModel):
    """Configuration for running policy comparison."""
    num_episodes: int = 10


@app.post("/api/comparison/run")
async def run_policy_comparison(config: ComparisonConfig):
    """
    Run comparison experiment across trained, heuristic, and random policies.
    
    This runs multiple episodes for each policy and returns statistical analysis.
    """
    try:
        # Run comparison (synchronously for now)
        results = run_comparison(
            num_episodes=config.num_episodes,
            trained_algo=trained_algo
        )
        
        # Convert dataclasses to dicts for JSON serialization
        return {
            "success": True,
            "comparison": {
                "trained": asdict(results.trained),
                "heuristic": asdict(results.heuristic),
                "random": asdict(results.random),
                "statistical_tests": {
                    "trained_vs_random_pvalue": results.trained_vs_random_pvalue,
                    "trained_vs_heuristic_pvalue": results.trained_vs_heuristic_pvalue,
                    "heuristic_vs_random_pvalue": results.heuristic_vs_random_pvalue
                },
                "improvements": {
                    "trained_over_random_percent": results.trained_improvement_over_random,
                    "trained_over_heuristic_percent": results.trained_improvement_over_heuristic
                }
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training-history")
async def get_training_history():
    """
    Get training history data for visualization.
    Returns iteration-by-iteration metrics from the last training run.
    """
    history_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models", "training_history.json"
    )
    
    # Also check in backend/models
    alt_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "backend", "models", "training_history.json"
    )
    
    # Try to find the history file
    for path in [history_path, alt_path]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    history = json.load(f)
                return {"success": True, "history": history}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading history: {str(e)}")
    
    # If no history file found, return a sample/mock history for demo
    return {
        "success": True,
        "history": {
            "metadata": {
                "start_time": "2024-01-01T00:00:00",
                "algorithm": "PPO",
                "num_agents": 5,
                "note": "No training history found. Run training to generate data."
            },
            "iterations": []
        },
        "message": "No training history found. Run training first."
    }


@app.get("/api/comparison/quick")
async def quick_comparison():
    """
    Run a quick comparison with fewer episodes (3 each).
    Good for testing the comparison feature.
    """
    try:
        results = run_comparison(
            num_episodes=3,
            trained_algo=trained_algo
        )
        
        # Return simplified results
        return {
            "success": True,
            "summary": {
                "trained": {
                    "reward": f"{results.trained.reward_mean:.1f} +/- {results.trained.reward_std:.1f}",
                    "stability": f"{results.trained.stability_mean * 100:.1f}%",
                    "grid_imports": f"{results.trained.grid_imports_mean:.1f} kWh",
                    "demand_satisfaction": f"{results.trained.demand_satisfaction_mean:.1f}%"
                },
                "heuristic": {
                    "reward": f"{results.heuristic.reward_mean:.1f} +/- {results.heuristic.reward_std:.1f}",
                    "stability": f"{results.heuristic.stability_mean * 100:.1f}%",
                    "grid_imports": f"{results.heuristic.grid_imports_mean:.1f} kWh",
                    "demand_satisfaction": f"{results.heuristic.demand_satisfaction_mean:.1f}%"
                },
                "random": {
                    "reward": f"{results.random.reward_mean:.1f} +/- {results.random.reward_std:.1f}",
                    "stability": f"{results.random.stability_mean * 100:.1f}%",
                    "grid_imports": f"{results.random.grid_imports_mean:.1f} kWh",
                    "demand_satisfaction": f"{results.random.demand_satisfaction_mean:.1f}%"
                }
            },
            "improvements": {
                "trained_vs_random": f"{results.trained_improvement_over_random:.1f}%",
                "trained_vs_heuristic": f"{results.trained_improvement_over_heuristic:.1f}%"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Global storage for last simulation results (for export)
last_simulation_results = {
    "daily_summaries": [],
    "overall_metrics": None,
    "timestamp": None
}


@app.post("/api/export/store-results")
async def store_simulation_results(data: dict):
    """Store simulation results for later export."""
    global last_simulation_results
    last_simulation_results = {
        "daily_summaries": data.get("daily_summaries", []),
        "overall_metrics": data.get("overall_metrics"),
        "timestamp": datetime.now().isoformat()
    }
    return {"success": True, "message": "Results stored for export"}


@app.get("/api/export/csv")
async def export_csv():
    """
    Export simulation results as CSV.
    Returns daily summaries in CSV format.
    """
    if not last_simulation_results["daily_summaries"]:
        raise HTTPException(status_code=404, detail="No simulation results available. Run a simulation first.")
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        "Day", "Total Reward", "Avg Stability (%)", "Grid Imports (kWh)",
        "Solar Utilization (%)", "Demand Satisfaction (%)", "Total Demand (kWh)",
        "Total Solar Generated (kWh)", "Total Solar Used (kWh)"
    ])
    
    # Write daily data
    for summary in last_simulation_results["daily_summaries"]:
        writer.writerow([
            summary.get("day", ""),
            round(summary.get("total_reward", 0), 2),
            round(summary.get("avg_stability", 0) * 100, 2),
            round(summary.get("grid_imports", 0), 2),
            round(summary.get("solar_utilization", 0), 2),
            round(summary.get("demand_satisfaction", 0), 2),
            round(summary.get("total_demand", 0), 2),
            round(summary.get("total_solar_generated", 0), 2),
            round(summary.get("total_solar_used", 0), 2)
        ])
    
    # Add blank row and overall metrics
    if last_simulation_results["overall_metrics"]:
        writer.writerow([])
        writer.writerow(["Overall Metrics"])
        metrics = last_simulation_results["overall_metrics"]
        writer.writerow(["Avg Daily Reward", round(metrics.get("avg_daily_reward", 0), 2)])
        writer.writerow(["Avg Stability (%)", round(metrics.get("avg_stability", 0) * 100, 2)])
        writer.writerow(["Avg Grid Imports (kWh)", round(metrics.get("avg_grid_imports", 0), 2)])
        writer.writerow(["Avg Solar Utilization (%)", round(metrics.get("avg_solar_utilization", 0), 2)])
        writer.writerow(["Avg Demand Satisfaction (%)", round(metrics.get("avg_demand_satisfaction", 0), 2)])
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=smart_grid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )


@app.get("/api/export/json")
async def export_json():
    """
    Export full simulation results as JSON.
    Includes all daily summaries, agent data, and overall metrics.
    """
    if not last_simulation_results["daily_summaries"]:
        raise HTTPException(status_code=404, detail="No simulation results available. Run a simulation first.")
    
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "simulation_timestamp": last_simulation_results["timestamp"],
        "config": {
            "num_agents": ENV_CONFIG["num_agents"],
            "max_energy_capacity": ENV_CONFIG["max_energy_capacity"],
            "max_steps_per_day": 288,
            "total_days": len(last_simulation_results["daily_summaries"])
        },
        "daily_summaries": last_simulation_results["daily_summaries"],
        "overall_metrics": last_simulation_results["overall_metrics"]
    }
    
    json_str = json.dumps(export_data, indent=2)
    
    return StreamingResponse(
        iter([json_str]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=smart_grid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
    )


@app.get("/api/scalability")
async def get_scalability_results():
    """
    Get scalability analysis results.
    Returns cached results if available, or runs a quick analysis.
    """
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models", "scalability_results.json"
    )
    
    # Also check alternate path
    alt_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "backend", "models", "scalability_results.json"
    )
    
    for path in [results_path, alt_path]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    results = json.load(f)
                return {"success": True, "results": results, "source": "cached"}
            except Exception as e:
                pass
    
    # No cached results - run quick analysis
    try:
        results = run_scalability_analysis(
            agent_counts=[3, 5, 7],
            episodes_per_count=2,
            save_path=results_path
        )
        return {"success": True, "results": asdict(results), "source": "generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scalability/run")
async def run_scalability_test(agent_counts: List[int] = [3, 5, 7, 10], episodes: int = 3):
    """
    Run a custom scalability analysis.
    """
    try:
        results = run_scalability_analysis(
            agent_counts=agent_counts,
            episodes_per_count=episodes
        )
        return {"success": True, "results": asdict(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/comparison-csv")
async def export_comparison_csv(num_episodes: int = 5):
    """
    Run a policy comparison and export results as CSV.
    """
    try:
        results = run_comparison(num_episodes=num_episodes, trained_algo=trained_algo)
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["Policy Comparison Results"])
        writer.writerow(["Generated", datetime.now().isoformat()])
        writer.writerow(["Episodes per Policy", num_episodes])
        writer.writerow([])
        
        # Results table
        writer.writerow(["Policy", "Reward Mean", "Reward Std", "Stability (%)", "Grid Imports (kWh)", 
                        "Solar Utilization (%)", "Demand Satisfaction (%)"])
        
        for policy_name, policy_data in [("PPO (Trained)", results.trained), 
                                          ("Heuristic", results.heuristic),
                                          ("Random", results.random)]:
            writer.writerow([
                policy_name,
                round(policy_data.reward_mean, 2),
                round(policy_data.reward_std, 2),
                round(policy_data.stability_mean * 100, 2),
                round(policy_data.grid_imports_mean, 2),
                round(policy_data.solar_utilization_mean, 2),
                round(policy_data.demand_satisfaction_mean, 2)
            ])
        
        # Statistical tests
        writer.writerow([])
        writer.writerow(["Statistical Significance (p-values)"])
        writer.writerow(["Trained vs Random", f"{results.trained_vs_random_pvalue:.6f}"])
        writer.writerow(["Trained vs Heuristic", f"{results.trained_vs_heuristic_pvalue:.6f}"])
        writer.writerow(["Heuristic vs Random", f"{results.heuristic_vs_random_pvalue:.6f}"])
        
        # Improvements
        writer.writerow([])
        writer.writerow(["Improvement Analysis"])
        writer.writerow(["Trained vs Random (%)", f"{results.trained_improvement_over_random:.2f}"])
        writer.writerow(["Trained vs Heuristic (%)", f"{results.trained_improvement_over_heuristic:.2f}"])
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=policy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/grid")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time grid state updates.
    
    Clients connect here to receive continuous updates during simulation.
    """
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Smart Grid MARL API",
            "num_agents": ENV_CONFIG["num_agents"]
        })
        
        # Keep connection alive and listen for messages
        while True:
            # Receive messages from client (ping/pong, commands, etc.)
            data = await websocket.receive_text()
            
            # Echo back for now (can add command handling later)
            await websocket.send_json({
                "type": "echo",
                "message": data
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    """Run on API startup."""
    print("="*60)
    print("Smart Grid MARL API Starting...")
    print("="*60)
    print(f"Environment: {ENV_CONFIG['num_agents']} agents")
    print(f"Max capacity: {ENV_CONFIG['max_energy_capacity']} kWh")
    
    # Try to load trained model
    if RAY_AVAILABLE:
        print("Loading trained model...")
        if load_trained_model():
            print("Using TRAINED PPO policy for inference")
        else:
            print("Using RANDOM policy (trained model not found)")
    else:
        print("Ray not available - using RANDOM policy")
    
    print(f"API documentation: http://localhost:8000/docs")
    print(f"WebSocket endpoint: ws://localhost:8000/ws/grid")
    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown."""
    global simulation_running
    simulation_running = False
    print("API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)







