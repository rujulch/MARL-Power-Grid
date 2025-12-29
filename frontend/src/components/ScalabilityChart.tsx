'use client';

import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, BarChart, Bar
} from 'recharts';
import { Scale, RefreshCw, Loader2, AlertTriangle, Users, Clock, Cpu, TrendingUp } from 'lucide-react';

interface AgentScaleResult {
  num_agents: number;
  episodes_run: number;
  mean_reward: number;
  std_reward: number;
  mean_stability: number;
  std_stability: number;
  total_time_seconds: number;
  avg_episode_time_seconds: number;
  avg_step_time_ms: number;
  memory_usage_mb: number;
  observation_dim: number;
  action_dim: number;
}

interface ScalabilityData {
  timestamp: string;
  agent_counts: number[];
  episodes_per_count: number;
  results: AgentScaleResult[];
  time_scaling_factor: number;
  reward_scaling_factor: number;
}

interface ScalabilityChartProps {
  apiUrl?: string;
}

export default function ScalabilityChart({ apiUrl = 'http://localhost:8000' }: ScalabilityChartProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ScalabilityData | null>(null);
  const [source, setSource] = useState<string>('');

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiUrl}/api/scalability`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const result = await response.json();
      if (result.success) {
        setData(result.results);
        setSource(result.source);
      } else {
        throw new Error(result.message || 'Failed to fetch scalability data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scalability analysis');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Prepare chart data
  const getChartData = () => {
    if (!data?.results) return [];
    return data.results.map(r => ({
      agents: r.num_agents,
      reward: r.mean_reward,
      rewardStd: r.std_reward,
      stability: r.mean_stability * 100,
      time: r.avg_episode_time_seconds,
      stepTime: r.avg_step_time_ms,
      memory: r.memory_usage_mb,
      obsDim: r.observation_dim,
      actDim: r.action_dim
    }));
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur-sm rounded-xl border border-slate-700/50 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Scale className="w-6 h-6 text-violet-400" />
              Scalability Analysis
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Performance metrics across different agent counts
            </p>
          </div>
          
          <button
            onClick={fetchData}
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
          <Loader2 className="w-12 h-12 text-violet-400 mx-auto animate-spin" />
          <p className="text-slate-400 mt-4">Running scalability analysis...</p>
          <p className="text-slate-500 text-sm mt-1">This may take a minute</p>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="p-8 text-center">
          <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
          <p className="text-amber-400">{error}</p>
          <button
            onClick={fetchData}
            className="mt-4 text-violet-400 hover:text-violet-300 text-sm"
          >
            Try again
          </button>
        </div>
      )}

      {/* Results */}
      {!loading && !error && data && (
        <div className="p-5 space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <Users className="w-4 h-4 text-slate-400" />
                <span className="text-xs text-slate-400">Agent Range</span>
              </div>
              <span className="text-lg font-semibold text-white">
                {data.agent_counts[0]} - {data.agent_counts[data.agent_counts.length - 1]}
              </span>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <Cpu className="w-4 h-4 text-slate-400" />
                <span className="text-xs text-slate-400">Episodes/Count</span>
              </div>
              <span className="text-lg font-semibold text-white">
                {data.episodes_per_count}
              </span>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="w-4 h-4 text-violet-400" />
                <span className="text-xs text-slate-400">Time Scaling</span>
              </div>
              <span className={`text-lg font-semibold ${
                data.time_scaling_factor < 0.5 ? 'text-emerald-400' : 
                data.time_scaling_factor < 1 ? 'text-amber-400' : 'text-red-400'
              }`}>
                {data.time_scaling_factor.toFixed(3)}x
              </span>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-slate-400">Reward Scaling</span>
              </div>
              <span className={`text-lg font-semibold ${
                data.reward_scaling_factor > 0 ? 'text-emerald-400' : 
                data.reward_scaling_factor > -0.2 ? 'text-amber-400' : 'text-red-400'
              }`}>
                {data.reward_scaling_factor > 0 ? '+' : ''}{(data.reward_scaling_factor * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-slate-400">
                  <th className="py-3 px-4 text-left">Agents</th>
                  <th className="py-3 px-4 text-right">Reward (mean +/- std)</th>
                  <th className="py-3 px-4 text-right">Stability</th>
                  <th className="py-3 px-4 text-right">Ep. Time (s)</th>
                  <th className="py-3 px-4 text-right">Step Time (ms)</th>
                  <th className="py-3 px-4 text-right">Obs Dim</th>
                  <th className="py-3 px-4 text-right">Act Dim</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {data.results.map((result, idx) => (
                  <tr key={result.num_agents} className="hover:bg-slate-800/30">
                    <td className="py-3 px-4">
                      <span className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded bg-violet-500/20 flex items-center justify-center text-violet-400 font-bold text-xs">
                          {result.num_agents}
                        </div>
                        {result.num_agents} agents
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-emerald-400">
                      {result.mean_reward.toFixed(1)} 
                      <span className="text-slate-500 ml-1">+/- {result.std_reward.toFixed(1)}</span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-cyan-400">
                      {(result.mean_stability * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-amber-400">
                      {result.avg_episode_time_seconds.toFixed(2)}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-slate-300">
                      {result.avg_step_time_ms.toFixed(2)}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-slate-400">
                      {result.observation_dim}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-slate-400">
                      {result.action_dim}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-2 gap-6">
            {/* Reward vs Agents */}
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <h3 className="text-sm font-medium text-slate-300 mb-4">Reward vs Number of Agents</h3>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={getChartData()} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="agents" 
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      label={{ value: 'Agents', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                    />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#f8fafc' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="reward" 
                      name="Mean Reward"
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={{ fill: '#10b981', r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Time vs Agents */}
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <h3 className="text-sm font-medium text-slate-300 mb-4">Episode Time vs Number of Agents</h3>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getChartData()} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="agents" 
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      label={{ value: 'Agents', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                    />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#f8fafc' }}
                    />
                    <Bar dataKey="time" name="Episode Time (s)" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Interpretation */}
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <h3 className="text-sm font-medium text-slate-300 mb-2">Interpretation</h3>
            <div className="text-sm text-slate-400 space-y-2">
              <p>
                <strong className="text-slate-200">Time Scaling Factor ({data.time_scaling_factor.toFixed(3)}x):</strong>{' '}
                {data.time_scaling_factor < 0.5 
                  ? 'Excellent! Time scales sub-linearly with agents.'
                  : data.time_scaling_factor < 1 
                  ? 'Good. Time scales linearly with agents.'
                  : 'Warning: Time scales super-linearly. Consider optimization.'}
              </p>
              <p>
                <strong className="text-slate-200">Reward Scaling ({(data.reward_scaling_factor * 100).toFixed(1)}%):</strong>{' '}
                {data.reward_scaling_factor > 0.1 
                  ? 'Positive scaling suggests cooperative benefits with more agents.'
                  : data.reward_scaling_factor > -0.1 
                  ? 'Stable performance across agent counts.'
                  : 'Performance decreases with more agents. May need algorithm tuning.'}
              </p>
            </div>
          </div>

          {/* Footer */}
          <div className="text-xs text-slate-500 text-center pt-2 border-t border-slate-700/50">
            {source === 'cached' ? 'Showing cached results' : 'Freshly generated results'}{' '}
            | Generated: {new Date(data.timestamp).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
}

