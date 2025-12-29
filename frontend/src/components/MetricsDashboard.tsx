'use client';

import React from 'react';
import { GridMetrics } from '@/types/grid';
import { Battery, Sun, Zap, TrendingUp } from 'lucide-react';
import { formatNumber } from '@/lib/utils';

interface MetricsDashboardProps {
  metrics: GridMetrics;
  stability: number;
}

export default function MetricsDashboard({ metrics, stability }: MetricsDashboardProps) {
  const metricsData = [
    {
      label: 'Avg Storage',
      value: `${formatNumber(metrics.mean_energy, 0)}`,
      unit: 'kWh',
      icon: Battery,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-400/10'
    },
    {
      label: 'Solar Gen',
      value: `${formatNumber(metrics.total_generation, 1)}`,
      unit: 'kW',
      icon: Sun,
      color: 'text-amber-400',
      bgColor: 'bg-amber-400/10'
    },
    {
      label: 'Demand',
      value: `${formatNumber(metrics.total_demand, 1)}`,
      unit: 'kW',
      icon: Zap,
      color: 'text-rose-400',
      bgColor: 'bg-rose-400/10'
    },
    {
      label: 'Reward',
      value: formatNumber(metrics.mean_reward, 1),
      unit: '/step',
      icon: TrendingUp,
      color: metrics.mean_reward >= 0 ? 'text-emerald-400' : 'text-red-400',
      bgColor: metrics.mean_reward >= 0 ? 'bg-emerald-400/10' : 'bg-red-400/10'
    },
  ];

  return (
    <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-3">
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
        Live Metrics
      </h3>
      <div className="grid grid-cols-2 gap-2">
        {metricsData.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <div
              key={index}
              className="bg-slate-900/50 rounded-lg p-2.5 border border-slate-700/50"
            >
              <div className="flex items-center gap-2 mb-1">
                <div className={`${metric.bgColor} p-1 rounded`}>
                  <Icon className={`w-3 h-3 ${metric.color}`} />
                </div>
                <span className="text-[10px] text-slate-500 uppercase">{metric.label}</span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className={`text-lg font-bold font-mono ${metric.color}`}>
                  {metric.value}
                </span>
                <span className="text-[10px] text-slate-500">{metric.unit}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}







