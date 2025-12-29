'use client';

import React from 'react';
import { AgentState } from '@/types/grid';
import { Battery, Zap, TrendingUp, Sun } from 'lucide-react';
import { cn, formatNumber, getEnergyColor, calculatePercentage, getNeighborhoodDisplayName, getNeighborhoodProfile } from '@/lib/utils';

interface AgentCardProps {
  agent: AgentState;
  maxCapacity?: number;
}

export default function AgentCard({ agent, maxCapacity = 100 }: AgentCardProps) {
  const energyPercentage = calculatePercentage(agent.energy_level, maxCapacity);
  const energyColor = getEnergyColor(agent.energy_level, maxCapacity);
  const netEnergy = agent.generation - agent.demand;
  const profile = getNeighborhoodProfile(agent.id);
  const displayName = getNeighborhoodDisplayName(agent.id);

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-700/50 p-3 hover:border-slate-600/50 transition-all">
      {/* Header with Energy Bar */}
      <div className="flex items-center gap-3 mb-2">
        <div 
          className="w-8 h-8 rounded-lg flex items-center justify-center text-sm"
          style={{ backgroundColor: `${profile?.color || energyColor}20` }}
        >
          {profile?.icon || agent.id.replace('agent_', '')}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-slate-300 truncate">
              {displayName}
            </span>
            <span className="text-xs font-mono text-slate-400">
              {formatNumber(agent.energy_level, 0)}/{maxCapacity}
            </span>
          </div>
          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full transition-all duration-300 rounded-full"
              style={{
                width: `${Math.min(energyPercentage, 100)}%`,
                backgroundColor: energyColor
              }}
            />
          </div>
        </div>
      </div>

      {/* Compact Metrics Row */}
      <div className="flex items-center justify-between text-[10px] text-slate-400">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <Sun className="w-3 h-3 text-amber-400" />
            <span className="text-slate-300">{formatNumber(agent.generation, 1)}</span>
          </span>
          <span className="flex items-center gap-1">
            <Zap className="w-3 h-3 text-rose-400" />
            <span className="text-slate-300">{formatNumber(agent.demand, 1)}</span>
          </span>
        </div>
        <div className={cn(
          "flex items-center gap-1 font-mono",
          netEnergy >= 0 ? "text-emerald-400" : "text-rose-400"
        )}>
          {netEnergy >= 0 ? '+' : ''}{formatNumber(netEnergy, 1)} net
        </div>
      </div>

      {/* Cumulative Reward (smaller) */}
      {agent.cumulative_reward !== undefined && (
        <div className="mt-2 pt-2 border-t border-slate-800 flex items-center justify-between">
          <span className="text-[10px] text-slate-500">Total Reward</span>
          <span className={cn(
            "text-xs font-mono font-medium",
            (agent.cumulative_reward ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
          )}>
            {formatNumber(agent.cumulative_reward ?? 0, 0)}
          </span>
        </div>
      )}
    </div>
  );
}







