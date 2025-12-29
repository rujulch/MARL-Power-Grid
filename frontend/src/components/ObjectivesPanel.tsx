'use client';

import React from 'react';
import { Target, Zap, Sun, Battery, TrendingUp, Clock, Calendar, Info, AlertTriangle } from 'lucide-react';
import { GridMetrics, SimulationTime } from '@/types/grid';

interface ObjectivesPanelProps {
  simulationTime?: SimulationTime;
  progress?: number;
  dayProgress?: number;
  step: number;
  totalSteps: number;
  currentDay?: number;
  totalDays?: number;
  usingTrainedModel?: boolean | null;
  metrics?: GridMetrics;
  completedDays?: number;
  stability?: number;
}

export default function ObjectivesPanel({
  simulationTime,
  progress = 0,
  dayProgress = 0,
  step,
  totalSteps,
  currentDay = 1,
  totalDays = 7,
  usingTrainedModel = null,
  metrics,
  completedDays = 0,
  stability = 0
}: ObjectivesPanelProps) {
  
  const objectives = [
    {
      icon: Battery,
      title: "Minimize Grid Imports",
      description: "Reduce expensive electricity purchases from the main grid",
      metric: metrics?.grid_imports !== undefined ? `${metrics.grid_imports.toFixed(1)} kWh imported` : "—",
      status: (metrics?.grid_imports ?? 100) < 50 ? 'good' : (metrics?.grid_imports ?? 100) < 100 ? 'medium' : 'poor'
    },
    {
      icon: Sun,
      title: "Maximize Solar Usage",
      description: "Use locally generated solar power before it's wasted",
      metric: metrics?.solar_utilization !== undefined ? `${metrics.solar_utilization.toFixed(1)}% utilized` : "—",
      status: (metrics?.solar_utilization ?? 0) > 80 ? 'good' : (metrics?.solar_utilization ?? 0) > 50 ? 'medium' : 'poor'
    },
    {
      icon: Zap,
      title: "Meet All Demands",
      description: "Keep all neighborhoods powered without blackouts",
      metric: metrics?.demand_satisfaction !== undefined ? `${metrics.demand_satisfaction.toFixed(1)}% satisfied` : "—",
      status: (metrics?.demand_satisfaction ?? 0) > 95 ? 'good' : (metrics?.demand_satisfaction ?? 0) > 80 ? 'medium' : 'poor'
    },
    {
      icon: TrendingUp,
      title: "Maximize Efficiency",
      description: "Coordinate between agents for optimal energy distribution",
      metric: metrics?.cumulative_reward !== undefined ? `${metrics.cumulative_reward.toFixed(0)} day reward` : "—",
      status: (metrics?.cumulative_reward ?? 0) > 5000 ? 'good' : (metrics?.cumulative_reward ?? 0) > 2000 ? 'medium' : 'poor'
    }
  ];

  const statusColors = {
    good: 'text-emerald-400 bg-emerald-400/10',
    medium: 'text-amber-400 bg-amber-400/10',
    poor: 'text-red-400 bg-red-400/10'
  };

  const getPeriodIcon = (period?: string) => {
    switch (period) {
      case 'Morning': return '🌅';
      case 'Afternoon': return '☀️';
      case 'Evening': return '🌆';
      case 'Night': return '🌙';
      default: return '⏱️';
    }
  };

  // Get stability status and explanation
  const getStabilityInfo = (stabilityValue: number) => {
    const pct = stabilityValue * 100;
    if (pct >= 80) {
      return {
        label: 'Excellent',
        color: 'text-emerald-400',
        bgColor: 'bg-emerald-400/20',
        borderColor: 'border-emerald-400/30',
        explanation: 'Optimal balance achieved - agents cooperating efficiently'
      };
    } else if (pct >= 60) {
      return {
        label: 'Good',
        color: 'text-cyan-400',
        bgColor: 'bg-cyan-400/20',
        borderColor: 'border-cyan-400/30',
        explanation: 'Normal fluctuations - agents handling solar/demand cycles well'
      };
    } else if (pct >= 40) {
      return {
        label: 'Fair',
        color: 'text-amber-400',
        bgColor: 'bg-amber-400/20',
        borderColor: 'border-amber-400/30',
        explanation: 'Some imbalance - room for improvement in coordination'
      };
    } else {
      return {
        label: 'Poor',
        color: 'text-red-400',
        bgColor: 'bg-red-400/20',
        borderColor: 'border-red-400/30',
        explanation: 'High instability - potential blackout risk'
      };
    }
  };

  const stabilityInfo = getStabilityInfo(stability);

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-5">
      {/* Header with Time */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-white">Simulation Objectives</h2>
        </div>
        
        {/* Model Status Badge - only show when simulation has provided actual status */}
        {usingTrainedModel !== null && (
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${
            usingTrainedModel 
              ? 'bg-emerald-400/20 text-emerald-400 border border-emerald-400/30' 
              : 'bg-amber-400/20 text-amber-400 border border-amber-400/30'
          }`}>
            {usingTrainedModel ? 'Trained Model' : 'Random Policy'}
          </div>
        )}
      </div>

      {/* Fallback Warning Banner - only show when explicitly using random (not before simulation starts) */}
      {usingTrainedModel === false && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 mb-4">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <span className="text-sm font-medium text-amber-400">Using Random Actions</span>
              <p className="text-xs text-amber-300/70 mt-1">
                Trained model inference failed. The simulation is running with random actions as a fallback.
                This demonstrates baseline behavior without intelligent coordination.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Grid Stability Panel with Explanation */}
      <div className={`rounded-lg p-4 mb-4 border ${stabilityInfo.bgColor} ${stabilityInfo.borderColor}`}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Battery className={`w-5 h-5 ${stabilityInfo.color}`} />
            <span className="text-sm font-medium text-slate-200">Grid Stability</span>
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-2xl font-mono font-bold ${stabilityInfo.color}`}>
              {(stability * 100).toFixed(1)}%
            </span>
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${stabilityInfo.bgColor} ${stabilityInfo.color}`}>
              {stabilityInfo.label}
            </span>
          </div>
        </div>
        
        {/* Stability bar */}
        <div className="h-2 bg-slate-900/50 rounded-full overflow-hidden mb-2">
          <div 
            className={`h-full transition-all duration-500 rounded-full ${
              stability >= 0.8 ? 'bg-emerald-500' :
              stability >= 0.6 ? 'bg-cyan-500' :
              stability >= 0.4 ? 'bg-amber-500' : 'bg-red-500'
            }`}
            style={{ width: `${stability * 100}%` }}
          />
        </div>
        
        <p className="text-xs text-slate-400">{stabilityInfo.explanation}</p>
        
        {/* Stability Scale */}
        <div className="mt-3 pt-3 border-t border-slate-700/50">
          <div className="flex items-center gap-1 mb-2">
            <Info className="w-3 h-3 text-slate-500" />
            <span className="text-xs text-slate-500">Stability Scale</span>
          </div>
          <div className="grid grid-cols-4 gap-1 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
              <span className="text-slate-400">80-100%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-cyan-500"></div>
              <span className="text-slate-400">60-80%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-amber-500"></div>
              <span className="text-slate-400">40-60%</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              <span className="text-slate-400">&lt;40%</span>
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            100% is unrealistic due to solar/demand cycles. 60-70% indicates successful management of real-world trade-offs.
          </p>
        </div>
      </div>

      {/* Multi-Day Progress */}
      <div className="bg-slate-900/50 rounded-lg p-4 mb-4 border border-slate-600/50">
        {/* Day indicators */}
        <div className="flex items-center gap-2 mb-3">
          <Calendar className="w-4 h-4 text-slate-400" />
          <span className="text-xs text-slate-400">7-Day Simulation</span>
          <div className="flex-1 flex items-center justify-end gap-1">
            {Array.from({ length: totalDays }, (_, i) => (
              <div
                key={i}
                className={`w-6 h-6 rounded flex items-center justify-center text-xs font-bold transition-all ${
                  i + 1 < currentDay 
                    ? 'bg-emerald-500/30 text-emerald-400 border border-emerald-500/50' 
                    : i + 1 === currentDay
                    ? 'bg-cyan-500/30 text-cyan-400 border border-cyan-500/50 animate-pulse'
                    : 'bg-slate-700/50 text-slate-500 border border-slate-600/50'
                }`}
              >
                {i + 1}
              </div>
            ))}
          </div>
        </div>

        {/* Current Time Display */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Clock className="w-5 h-5 text-slate-400" />
            <div>
              <div className="text-2xl font-mono font-bold text-white">
                {simulationTime ? (
                  <>
                    <span className="text-cyan-400">Day {currentDay}</span>
                    <span className="text-slate-500 mx-2">|</span>
                    {simulationTime.time_string}
                  </>
                ) : (
                  <span className="text-slate-500">--:-- --</span>
                )}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {simulationTime?.period && (
                  <span>{getPeriodIcon(simulationTime.period)} {simulationTime.period}</span>
                )}
                {' • '}
                Day {currentDay} of {totalDays}
              </div>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-sm text-slate-400">Overall</div>
            <div className="text-xl font-bold text-cyan-400">{progress.toFixed(0)}%</div>
          </div>
        </div>
        
        {/* Day Progress Bar */}
        <div className="mt-3">
          <div className="flex justify-between text-xs text-slate-500 mb-1">
            <span>Day {currentDay} Progress</span>
            <span>{dayProgress.toFixed(0)}%</span>
          </div>
          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-cyan-500 transition-all duration-300"
              style={{ width: `${dayProgress}%` }}
            />
          </div>
        </div>
        
        {/* Overall Progress Bar */}
        <div className="mt-2">
          <div className="flex justify-between text-xs text-slate-500 mb-1">
            <span>Total Progress</span>
            <span>{completedDays} / {totalDays} days</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Objectives Grid */}
      <div className="grid grid-cols-2 gap-3">
        {objectives.map((obj, idx) => {
          const Icon = obj.icon;
          return (
            <div 
              key={idx}
              className="bg-slate-900/30 rounded-lg p-3 border border-slate-700/50 hover:border-slate-600 transition-all"
            >
              <div className="flex items-start gap-2 mb-2">
                <div className={`p-1.5 rounded ${statusColors[obj.status]}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium text-white truncate">{obj.title}</h3>
                  <p className="text-xs text-slate-400 line-clamp-2">{obj.description}</p>
                </div>
              </div>
              <div className={`text-sm font-mono ${statusColors[obj.status].split(' ')[0]}`}>
                {obj.metric}
              </div>
            </div>
          );
        })}
      </div>

      {/* What's Happening Explanation */}
      <div className="mt-4 p-3 bg-slate-900/30 rounded-lg border border-slate-700/50">
        <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
          What&apos;s Happening?
        </h3>
        <p className="text-xs text-slate-400 leading-relaxed">
          <strong className="text-slate-300">5 AI agents</strong> (neighborhoods) are learning to 
          <strong className="text-emerald-400"> share energy efficiently</strong>. Each has solar panels 
          and batteries. They must <strong className="text-cyan-400">balance local demand</strong> while 
          minimizing expensive grid imports. Agents can <strong className="text-amber-400">trade energy</strong> with 
          neighbors when they have surplus or deficit.
        </p>
      </div>

      {/* Understanding Rewards */}
      <div className="mt-3 p-3 bg-slate-900/30 rounded-lg border border-slate-700/50">
        <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
          Understanding Rewards
        </h3>
        <p className="text-xs text-slate-400 leading-relaxed">
          Each agent earns rewards for: <strong className="text-cyan-400">using solar</strong> instead of grid power, 
          <strong className="text-emerald-400"> maintaining balanced batteries</strong> (not too full/empty), 
          <strong className="text-amber-400"> sharing with neighbors</strong>, and 
          <strong className="text-red-400"> avoiding blackouts</strong>. Higher cumulative rewards = better decisions.
        </p>
      </div>
    </div>
  );
}
