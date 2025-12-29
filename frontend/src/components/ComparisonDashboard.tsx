'use client';

import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Play, Loader2, Trophy, TrendingUp, Zap, Battery, Sun, AlertTriangle, CheckCircle } from 'lucide-react';

interface PolicyStats {
  policy_name: string;
  num_episodes: number;
  reward_mean: number;
  reward_std: number;
  reward_min: number;
  reward_max: number;
  stability_mean: number;
  stability_std: number;
  grid_imports_mean: number;
  grid_imports_std: number;
  solar_utilization_mean: number;
  solar_utilization_std: number;
  demand_satisfaction_mean: number;
  demand_satisfaction_std: number;
}

interface ComparisonData {
  trained: PolicyStats;
  heuristic: PolicyStats;
  random: PolicyStats;
  statistical_tests: {
    trained_vs_random_pvalue: number;
    trained_vs_heuristic_pvalue: number;
    heuristic_vs_random_pvalue: number;
  };
  improvements: {
    trained_over_random_percent: number;
    trained_over_heuristic_percent: number;
  };
}

interface ComparisonDashboardProps {
  apiUrl?: string;
}

const POLICY_COLORS = {
  trained: '#10b981',    // Emerald
  heuristic: '#f59e0b',  // Amber
  random: '#ef4444'      // Red
};

export default function ComparisonDashboard({ apiUrl = 'http://localhost:8000' }: ComparisonDashboardProps) {
  const [loading, setLoading] = useState(false);
  const [numEpisodes, setNumEpisodes] = useState(10);
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runComparison = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiUrl}/api/comparison/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_episodes: numEpisodes })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setComparison(data.comparison);
      } else {
        throw new Error(data.message || 'Comparison failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run comparison');
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data
  const getRewardChartData = () => {
    if (!comparison) return [];
    return [
      { name: 'PPO (Trained)', value: comparison.trained.reward_mean, std: comparison.trained.reward_std },
      { name: 'Heuristic', value: comparison.heuristic.reward_mean, std: comparison.heuristic.reward_std },
      { name: 'Random', value: comparison.random.reward_mean, std: comparison.random.reward_std }
    ];
  };

  const getMetricsChartData = () => {
    if (!comparison) return [];
    return [
      {
        metric: 'Stability',
        trained: comparison.trained.stability_mean * 100,
        heuristic: comparison.heuristic.stability_mean * 100,
        random: comparison.random.stability_mean * 100
      },
      {
        metric: 'Demand Sat.',
        trained: comparison.trained.demand_satisfaction_mean,
        heuristic: comparison.heuristic.demand_satisfaction_mean,
        random: comparison.random.demand_satisfaction_mean
      },
      {
        metric: 'Solar Util.',
        trained: comparison.trained.solar_utilization_mean,
        heuristic: comparison.heuristic.solar_utilization_mean,
        random: comparison.random.solar_utilization_mean
      }
    ];
  };

  const formatPValue = (p: number) => {
    if (p < 0.001) return 'p < 0.001';
    if (p < 0.01) return `p < 0.01`;
    if (p < 0.05) return `p < 0.05`;
    return `p = ${p.toFixed(3)}`;
  };

  const isSignificant = (p: number) => p < 0.05;

  return (
    <div className="bg-slate-900/80 backdrop-blur-sm rounded-xl border border-slate-700/50 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Trophy className="w-6 h-6 text-amber-400" />
              Policy Comparison
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Compare trained PPO model vs heuristic baseline vs random policy
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <label className="text-sm text-slate-400">Episodes:</label>
              <select
                value={numEpisodes}
                onChange={(e) => setNumEpisodes(Number(e.target.value))}
                disabled={loading}
                className="bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-cyan-500"
              >
                <option value={3}>3 (Quick)</option>
                <option value={5}>5</option>
                <option value={10}>10 (Standard)</option>
                <option value={20}>20 (Thorough)</option>
                <option value={50}>50 (Statistical)</option>
              </select>
            </div>
            
            <button
              onClick={runComparison}
              disabled={loading}
              className="flex items-center gap-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 
                       text-white px-4 py-2 rounded-lg font-medium transition-colors"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Comparison
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-500/10 border-b border-red-500/30">
          <div className="flex items-center gap-2 text-red-400">
            <AlertTriangle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="p-12 text-center">
          <Loader2 className="w-12 h-12 text-cyan-400 mx-auto animate-spin" />
          <p className="text-slate-400 mt-4">Running {numEpisodes} episodes for each policy...</p>
          <p className="text-slate-500 text-sm mt-1">This may take a few minutes</p>
        </div>
      )}

      {/* Results */}
      {comparison && !loading && (
        <div className="p-5 space-y-6">
          {/* Key Finding Banner */}
          {comparison.trained.policy_name.includes('unavailable') ? (
            <div className="bg-gradient-to-r from-amber-500/20 to-orange-500/20 rounded-lg p-4 border border-amber-500/30">
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-8 h-8 text-amber-400" />
                <div>
                  <h3 className="text-lg font-semibold text-white">
                    Trained model inference unavailable
                  </h3>
                  <p className="text-slate-300 text-sm">
                    Using heuristic policy as fallback for &quot;Trained&quot; column. 
                    This is due to Ray API compatibility issues.
                    The heuristic still outperforms random by{' '}
                    <span className="text-emerald-400 font-medium">
                      +{comparison.improvements.trained_over_random_percent.toFixed(1)}%
                    </span>
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 rounded-lg p-4 border border-emerald-500/30">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-8 h-8 text-emerald-400" />
                <div>
                  <h3 className="text-lg font-semibold text-white">
                    Trained model outperforms baselines
                  </h3>
                  <p className="text-slate-300 text-sm">
                    <span className="text-emerald-400 font-medium">
                      +{comparison.improvements.trained_over_random_percent.toFixed(1)}%
                    </span> improvement over random, {' '}
                    <span className="text-cyan-400 font-medium">
                      +{comparison.improvements.trained_over_heuristic_percent.toFixed(1)}%
                    </span> over heuristic
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Comparison Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="py-3 px-4 text-left text-slate-400 font-medium">Metric</th>
                  <th className="py-3 px-4 text-right">
                    <span className="flex items-center justify-end gap-2 text-emerald-400 font-medium">
                      <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                      {comparison.trained.policy_name.includes('unavailable') ? 'PPO*' : 'PPO (Trained)'}
                    </span>
                  </th>
                  <th className="py-3 px-4 text-right">
                    <span className="flex items-center justify-end gap-2 text-amber-400 font-medium">
                      <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                      Heuristic
                    </span>
                  </th>
                  <th className="py-3 px-4 text-right">
                    <span className="flex items-center justify-end gap-2 text-red-400 font-medium">
                      <div className="w-3 h-3 rounded-full bg-red-500"></div>
                      Random
                    </span>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                <tr>
                  <td className="py-3 px-4 text-slate-300 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-slate-500" />
                    Total Reward
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-emerald-400">
                    {comparison.trained.reward_mean.toFixed(1)} <span className="text-slate-500">+/- {comparison.trained.reward_std.toFixed(1)}</span>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-amber-400">
                    {comparison.heuristic.reward_mean.toFixed(1)} <span className="text-slate-500">+/- {comparison.heuristic.reward_std.toFixed(1)}</span>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-red-400">
                    {comparison.random.reward_mean.toFixed(1)} <span className="text-slate-500">+/- {comparison.random.reward_std.toFixed(1)}</span>
                  </td>
                </tr>
                <tr>
                  <td className="py-3 px-4 text-slate-300 flex items-center gap-2">
                    <Battery className="w-4 h-4 text-slate-500" />
                    Grid Stability
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-emerald-400">
                    {(comparison.trained.stability_mean * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-amber-400">
                    {(comparison.heuristic.stability_mean * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-red-400">
                    {(comparison.random.stability_mean * 100).toFixed(1)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-3 px-4 text-slate-300 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-slate-500" />
                    Grid Imports (kWh)
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-emerald-400">
                    {comparison.trained.grid_imports_mean.toFixed(1)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-amber-400">
                    {comparison.heuristic.grid_imports_mean.toFixed(1)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-red-400">
                    {comparison.random.grid_imports_mean.toFixed(1)}
                  </td>
                </tr>
                <tr>
                  <td className="py-3 px-4 text-slate-300 flex items-center gap-2">
                    <Sun className="w-4 h-4 text-slate-500" />
                    Solar Utilization
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-emerald-400">
                    {comparison.trained.solar_utilization_mean.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-amber-400">
                    {comparison.heuristic.solar_utilization_mean.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-red-400">
                    {comparison.random.solar_utilization_mean.toFixed(1)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-3 px-4 text-slate-300 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-slate-500" />
                    Demand Satisfaction
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-emerald-400">
                    {comparison.trained.demand_satisfaction_mean.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-amber-400">
                    {comparison.heuristic.demand_satisfaction_mean.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-red-400">
                    {comparison.random.demand_satisfaction_mean.toFixed(1)}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-2 gap-6">
            {/* Reward Comparison Bar Chart */}
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <h3 className="text-sm font-medium text-slate-300 mb-4">Total Reward by Policy</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getRewardChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#f8fafc' }}
                    />
                    <Bar dataKey="value" name="Mean Reward">
                      {getRewardChartData().map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={index === 0 ? POLICY_COLORS.trained : index === 1 ? POLICY_COLORS.heuristic : POLICY_COLORS.random} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Metrics Comparison Grouped Bar Chart */}
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <h3 className="text-sm font-medium text-slate-300 mb-4">Performance Metrics (%)</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getMetricsChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#f8fafc' }}
                    />
                    <Legend />
                    <Bar dataKey="trained" name="PPO" fill={POLICY_COLORS.trained} />
                    <Bar dataKey="heuristic" name="Heuristic" fill={POLICY_COLORS.heuristic} />
                    <Bar dataKey="random" name="Random" fill={POLICY_COLORS.random} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Statistical Significance */}
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Statistical Significance (t-test)</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className={`p-3 rounded-lg border ${
                isSignificant(comparison.statistical_tests.trained_vs_random_pvalue) 
                  ? 'bg-emerald-500/10 border-emerald-500/30' 
                  : 'bg-slate-700/30 border-slate-600/30'
              }`}>
                <div className="text-xs text-slate-400 mb-1">Trained vs Random</div>
                <div className={`font-mono text-sm ${
                  isSignificant(comparison.statistical_tests.trained_vs_random_pvalue) 
                    ? 'text-emerald-400' 
                    : 'text-slate-400'
                }`}>
                  {formatPValue(comparison.statistical_tests.trained_vs_random_pvalue)}
                  {isSignificant(comparison.statistical_tests.trained_vs_random_pvalue) && (
                    <span className="ml-2 text-emerald-400">Significant</span>
                  )}
                </div>
              </div>
              <div className={`p-3 rounded-lg border ${
                isSignificant(comparison.statistical_tests.trained_vs_heuristic_pvalue) 
                  ? 'bg-emerald-500/10 border-emerald-500/30' 
                  : 'bg-slate-700/30 border-slate-600/30'
              }`}>
                <div className="text-xs text-slate-400 mb-1">Trained vs Heuristic</div>
                <div className={`font-mono text-sm ${
                  isSignificant(comparison.statistical_tests.trained_vs_heuristic_pvalue) 
                    ? 'text-emerald-400' 
                    : 'text-slate-400'
                }`}>
                  {formatPValue(comparison.statistical_tests.trained_vs_heuristic_pvalue)}
                  {isSignificant(comparison.statistical_tests.trained_vs_heuristic_pvalue) && (
                    <span className="ml-2 text-emerald-400">Significant</span>
                  )}
                </div>
              </div>
              <div className={`p-3 rounded-lg border ${
                isSignificant(comparison.statistical_tests.heuristic_vs_random_pvalue) 
                  ? 'bg-amber-500/10 border-amber-500/30' 
                  : 'bg-slate-700/30 border-slate-600/30'
              }`}>
                <div className="text-xs text-slate-400 mb-1">Heuristic vs Random</div>
                <div className={`font-mono text-sm ${
                  isSignificant(comparison.statistical_tests.heuristic_vs_random_pvalue) 
                    ? 'text-amber-400' 
                    : 'text-slate-400'
                }`}>
                  {formatPValue(comparison.statistical_tests.heuristic_vs_random_pvalue)}
                  {isSignificant(comparison.statistical_tests.heuristic_vs_random_pvalue) && (
                    <span className="ml-2 text-amber-400">Significant</span>
                  )}
                </div>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-3">
              p &lt; 0.05 indicates the difference is statistically significant (not due to random chance)
            </p>
          </div>

          {/* Episodes Summary */}
          <div className="text-xs text-slate-500 text-center pt-2 border-t border-slate-700/50">
            Results based on {comparison.trained.num_episodes} episodes per policy
          </div>
        </div>
      )}

      {/* Empty State */}
      {!comparison && !loading && !error && (
        <div className="p-12 text-center">
          <Trophy className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-slate-300 mb-2">No Comparison Results Yet</h3>
          <p className="text-slate-500 text-sm max-w-md mx-auto">
            Click &quot;Run Comparison&quot; to compare the trained PPO model against heuristic and random baseline policies.
            This will run multiple episodes and provide statistical analysis.
          </p>
        </div>
      )}
    </div>
  );
}

