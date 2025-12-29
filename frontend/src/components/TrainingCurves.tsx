'use client';

import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, Area, AreaChart, ReferenceLine
} from 'recharts';
import { TrendingUp, RefreshCw, Loader2, AlertTriangle, Award, Clock, Cpu } from 'lucide-react';

interface TrainingIteration {
  iteration: number;
  timestamp: string;
  episode_reward_mean: number;
  episode_reward_min: number;
  episode_reward_max: number;
  episode_len_mean: number;
  episodes_total: number;
  timesteps_total: number;
  best_reward_so_far: number;
  policy_rewards: Record<string, number>;
}

interface TrainingMetadata {
  start_time: string;
  end_time?: string;
  algorithm: string;
  num_agents: number;
  total_iterations?: number;
  final_reward?: number;
  best_reward?: number;
  best_iteration?: number;
  config?: Record<string, any>;
}

interface TrainingHistory {
  metadata: TrainingMetadata;
  iterations: TrainingIteration[];
}

interface TrainingCurvesProps {
  apiUrl?: string;
}

export default function TrainingCurves({ apiUrl = 'http://localhost:8000' }: TrainingCurvesProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<TrainingHistory | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiUrl}/api/training-history`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setHistory(data.history);
      } else {
        throw new Error(data.message || 'Failed to fetch training history');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load training history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // Prepare chart data
  const getRewardChartData = () => {
    if (!history?.iterations) return [];
    return history.iterations.map(iter => ({
      iteration: iter.iteration,
      mean: iter.episode_reward_mean,
      min: iter.episode_reward_min,
      max: iter.episode_reward_max,
      best: iter.best_reward_so_far
    }));
  };

  const getEpisodeLengthData = () => {
    if (!history?.iterations) return [];
    return history.iterations.map(iter => ({
      iteration: iter.iteration,
      length: iter.episode_len_mean,
      episodes: iter.episodes_total,
      timesteps: iter.timesteps_total
    }));
  };

  // Calculate convergence metrics
  const getConvergenceInfo = () => {
    if (!history?.iterations || history.iterations.length < 10) {
      return { converged: false, convergenceIteration: null, improvementRate: null };
    }

    const rewards = history.iterations.map(i => i.episode_reward_mean);
    const last10 = rewards.slice(-10);
    const first10 = rewards.slice(0, 10);
    
    const last10Avg = last10.reduce((a, b) => a + b, 0) / last10.length;
    const first10Avg = first10.reduce((a, b) => a + b, 0) / first10.length;
    const last10Std = Math.sqrt(last10.map(x => Math.pow(x - last10Avg, 2)).reduce((a, b) => a + b, 0) / last10.length);
    
    // Check if converged (low variance in last 10 iterations)
    const converged = last10Std < Math.abs(last10Avg) * 0.1;
    
    // Find approximate convergence point (where reward plateaus)
    let convergenceIteration = null;
    for (let i = 10; i < rewards.length; i++) {
      const windowAvg = rewards.slice(i - 10, i).reduce((a, b) => a + b, 0) / 10;
      if (windowAvg > last10Avg * 0.9) {
        convergenceIteration = i;
        break;
      }
    }

    const improvementRate = ((last10Avg - first10Avg) / Math.abs(first10Avg) * 100);

    return { converged, convergenceIteration, improvementRate, last10Avg, last10Std };
  };

  const formatDuration = (start: string, end?: string) => {
    if (!end) return 'In progress';
    const startDate = new Date(start);
    const endDate = new Date(end);
    const durationMs = endDate.getTime() - startDate.getTime();
    const minutes = Math.floor(durationMs / 60000);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    return `${minutes}m`;
  };

  const convergenceInfo = getConvergenceInfo();

  return (
    <div className="bg-slate-900/80 backdrop-blur-sm rounded-xl border border-slate-700/50 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-cyan-400" />
              Training Progress
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Learning curves and convergence analysis
            </p>
          </div>
          
          <button
            onClick={fetchHistory}
            disabled={loading}
            className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-3 py-1.5 rounded-lg text-sm transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="p-12 text-center">
          <Loader2 className="w-12 h-12 text-cyan-400 mx-auto animate-spin" />
          <p className="text-slate-400 mt-4">Loading training history...</p>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="p-8 text-center">
          <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
          <p className="text-amber-400">{error}</p>
          <button
            onClick={fetchHistory}
            className="mt-4 text-cyan-400 hover:text-cyan-300 text-sm"
          >
            Try again
          </button>
        </div>
      )}

      {/* No Data State */}
      {!loading && !error && history?.iterations?.length === 0 && (
        <div className="p-12 text-center">
          <TrendingUp className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-slate-300 mb-2">No Training Data Yet</h3>
          <p className="text-slate-500 text-sm max-w-md mx-auto">
            Training history will appear here after you run the training script.
            The curves show how the model improves over iterations.
          </p>
        </div>
      )}

      {/* Training Data Display */}
      {!loading && !error && history?.iterations && history.iterations.length > 0 && (
        <div className="p-5 space-y-6">
          {/* Metadata Summary */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <Cpu className="w-4 h-4 text-slate-400" />
                <span className="text-xs text-slate-400">Algorithm</span>
              </div>
              <span className="text-lg font-semibold text-white">{history.metadata.algorithm}</span>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4 text-slate-400" />
                <span className="text-xs text-slate-400">Total Iterations</span>
              </div>
              <span className="text-lg font-semibold text-white">{history.metadata.total_iterations || history.iterations.length}</span>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <Award className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-slate-400">Best Reward</span>
              </div>
              <span className="text-lg font-semibold text-emerald-400">
                {history.metadata.best_reward?.toFixed(1) || 'N/A'}
              </span>
              {history.metadata.best_iteration && (
                <span className="text-xs text-slate-500 ml-2">@ iter {history.metadata.best_iteration}</span>
              )}
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="w-4 h-4 text-slate-400" />
                <span className="text-xs text-slate-400">Duration</span>
              </div>
              <span className="text-lg font-semibold text-white">
                {formatDuration(history.metadata.start_time, history.metadata.end_time)}
              </span>
            </div>
          </div>

          {/* Convergence Analysis */}
          {convergenceInfo.improvementRate !== null && (
            <div className={`rounded-lg p-4 border ${
              convergenceInfo.converged 
                ? 'bg-emerald-500/10 border-emerald-500/30' 
                : 'bg-cyan-500/10 border-cyan-500/30'
            }`}>
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-full ${
                  convergenceInfo.converged ? 'bg-emerald-500/20' : 'bg-cyan-500/20'
                }`}>
                  <TrendingUp className={`w-5 h-5 ${
                    convergenceInfo.converged ? 'text-emerald-400' : 'text-cyan-400'
                  }`} />
                </div>
                <div>
                  <h3 className={`font-semibold ${
                    convergenceInfo.converged ? 'text-emerald-400' : 'text-cyan-400'
                  }`}>
                    {convergenceInfo.converged ? 'Training Converged' : 'Training Progress'}
                  </h3>
                  <p className="text-slate-300 text-sm">
                    {convergenceInfo.improvementRate > 0 
                      ? `+${convergenceInfo.improvementRate.toFixed(1)}% improvement from start to end`
                      : `${convergenceInfo.improvementRate.toFixed(1)}% change from start to end`
                    }
                    {convergenceInfo.convergenceIteration && (
                      <span className="text-slate-400 ml-2">
                        (plateau around iteration {convergenceInfo.convergenceIteration})
                      </span>
                    )}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Reward Curve */}
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <h3 className="text-sm font-medium text-slate-300 mb-4">Episode Reward Over Training</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={getRewardChartData()} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="iteration" 
                    tick={{ fill: '#94a3b8', fontSize: 12 }} 
                    label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                  />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#f8fafc' }}
                    formatter={(value: number, name: string) => [value.toFixed(2), name]}
                  />
                  <Legend />
                  {history.metadata.best_reward && (
                    <ReferenceLine 
                      y={history.metadata.best_reward} 
                      stroke="#10b981" 
                      strokeDasharray="5 5"
                      label={{ value: 'Best', fill: '#10b981', fontSize: 10 }}
                    />
                  )}
                  <Area 
                    type="monotone" 
                    dataKey="mean" 
                    name="Mean Reward"
                    stroke="#06b6d4" 
                    fill="url(#rewardGradient)"
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="best" 
                    name="Best So Far"
                    stroke="#10b981" 
                    strokeWidth={1.5}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Episode Length Curve */}
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <h3 className="text-sm font-medium text-slate-300 mb-4">Episode Length & Timesteps</h3>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={getEpisodeLengthData()} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="iteration" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis yAxisId="left" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#f8fafc' }}
                  />
                  <Legend />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="length" 
                    name="Avg Episode Length"
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="timesteps" 
                    name="Total Timesteps"
                    stroke="#8b5cf6" 
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Training Info */}
          <div className="text-xs text-slate-500 text-center pt-2 border-t border-slate-700/50">
            Training started: {new Date(history.metadata.start_time).toLocaleString()}
            {history.metadata.end_time && (
              <span> | Completed: {new Date(history.metadata.end_time).toLocaleString()}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

