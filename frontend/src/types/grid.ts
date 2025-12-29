/**
 * TypeScript types for Smart Grid MARL system
 */

export interface AgentState {
  id: string;
  name?: string;
  energy_level: number;
  demand: number;
  generation: number;
  x: number;
  y: number;
  reward?: number;
  cumulative_reward?: number;
}

export interface EnergyFlow {
  source: string;
  target: string;
  amount: number;
  type: 'transfer' | 'request';
}

export interface GridMetrics {
  mean_energy: number;
  total_demand: number;
  total_generation: number;
  mean_reward: number;
  cumulative_reward?: number;
  grid_imports?: number;
  solar_utilization?: number;
  demand_satisfaction?: number;
}

export interface SimulationTime {
  day: number;
  hour: number;
  minute: number;
  time_string: string;
  period: 'Morning' | 'Afternoon' | 'Evening' | 'Night';
}

export interface GridState {
  type?: string;
  timestamp: string;
  step: number;
  steps_per_day?: number;
  current_day?: number;
  total_days?: number;
  global_step?: number;
  total_steps?: number;
  progress?: number;
  day_progress?: number;
  stability: number;
  using_trained_model?: boolean;
  simulation_time?: SimulationTime;
  agents: AgentState[];
  energy_flows: EnergyFlow[];
  metrics: GridMetrics;
}

export interface SimulationConfig {
  num_days?: number;
  steps_per_day?: number;
  scenario?: 'default' | 'high_demand' | 'high_solar';
}

export interface AgentInfo {
  id: string;
  name: string;
  max_capacity: number;
  max_transfer: number;
}

// Daily summary types for multi-day simulation
export interface DailyAgentStats {
  agent_id: string;
  name: string;
  reward: number;
  avg_battery_level: number;
  total_demand: number;
  solar_generated: number;
  grid_imported: number;
  energy_traded_in: number;
  energy_traded_out: number;
  battery_start: number;
  battery_end: number;
  demand_satisfaction: number;
}

export interface DailySummary {
  day: number;
  total_reward: number;
  avg_stability: number;
  grid_imports: number;
  total_solar_generated: number;
  total_solar_used: number;
  solar_utilization: number;
  demand_satisfaction: number;
  total_demand: number;
  agents: DailyAgentStats[];
}

export interface SimulationCompleteData {
  type: 'simulation_complete';
  message: string;
  total_days: number;
  daily_summaries: DailySummary[];
  overall_metrics: {
    avg_daily_reward: number;
    avg_stability: number;
    avg_grid_imports: number;
    avg_solar_utilization: number;
    avg_demand_satisfaction: number;
  };
}

export interface DayCompleteData {
  type: 'day_complete';
  day: number;
  total_days: number;
  summary: DailySummary;
  message: string;
}







