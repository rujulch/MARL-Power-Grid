'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface InfoPanelProps {
  className?: string;
}

interface InfoSection {
  title: string;
  content: React.ReactNode;
}

export default function InfoPanel({ className = '' }: InfoPanelProps) {
  const [activeSection, setActiveSection] = useState<string | null>('stability');

  const sections: Record<string, InfoSection> = {
    stability: {
      title: 'Grid Stability',
      content: (
        <div className="space-y-4">
          <p className="text-slate-300 text-sm leading-relaxed">
            Grid stability measures how well-balanced the energy distribution is across all neighborhoods. 
            It considers battery levels, variance between agents, and blackout prevention.
          </p>
          
          <div className="space-y-2">
            <h4 className="text-amber-400 font-semibold text-sm">Stability Scale</h4>
            <div className="space-y-1.5">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                <span className="text-emerald-400 font-medium text-sm">80-100%</span>
                <span className="text-slate-400 text-sm">Excellent - Optimal balance achieved</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-cyan-500"></div>
                <span className="text-cyan-400 font-medium text-sm">60-80%</span>
                <span className="text-slate-400 text-sm">Good - Normal fluctuations, agents cooperating</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                <span className="text-amber-400 font-medium text-sm">40-60%</span>
                <span className="text-slate-400 text-sm">Fair - Some imbalance, needs improvement</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <span className="text-red-400 font-medium text-sm">&lt;40%</span>
                <span className="text-slate-400 text-sm">Poor - High instability, blackout risk</span>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <h4 className="text-slate-200 font-medium text-sm mb-2">Why isn&apos;t 100% the goal?</h4>
            <p className="text-slate-400 text-xs leading-relaxed">
              Perfect 100% stability is unrealistic because solar generation peaks at noon but drops to zero at night, 
              while demand peaks in morning/evening. The agents must balance using energy efficiently versus 
              hoarding it for stability. A 60-70% stability indicates the model is successfully managing these 
              real-world trade-offs.
            </p>
          </div>
        </div>
      )
    },
    rewards: {
      title: 'Agent Rewards',
      content: (
        <div className="space-y-4">
          <p className="text-slate-300 text-sm leading-relaxed">
            Each agent (neighborhood) earns rewards based on how well they balance multiple competing objectives.
            Higher rewards indicate better decision-making.
          </p>

          <div className="space-y-2">
            <h4 className="text-amber-400 font-semibold text-sm">Reward Components</h4>
            <div className="space-y-2 text-sm">
              <div className="bg-slate-800/50 rounded p-2 border border-slate-700">
                <span className="text-cyan-400 font-medium">Efficiency Bonus</span>
                <p className="text-slate-400 text-xs mt-1">
                  Reward for using solar energy directly instead of importing from grid
                </p>
              </div>
              <div className="bg-slate-800/50 rounded p-2 border border-slate-700">
                <span className="text-emerald-400 font-medium">Stability Contribution</span>
                <p className="text-slate-400 text-xs mt-1">
                  Reward for maintaining balanced battery levels (not too full, not too empty)
                </p>
              </div>
              <div className="bg-slate-800/50 rounded p-2 border border-slate-700">
                <span className="text-amber-400 font-medium">Cooperation Bonus</span>
                <p className="text-slate-400 text-xs mt-1">
                  Reward for sharing energy with neighbors who need it
                </p>
              </div>
              <div className="bg-slate-800/50 rounded p-2 border border-slate-700">
                <span className="text-red-400 font-medium">Blackout Penalty</span>
                <p className="text-slate-400 text-xs mt-1">
                  Heavy penalty for letting battery drop below critical levels
                </p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <h4 className="text-slate-200 font-medium text-sm mb-2">Interpreting Rewards</h4>
            <p className="text-slate-400 text-xs leading-relaxed">
              Cumulative daily rewards above 200 indicate good performance. Agents with similar rewards 
              are cooperating effectively. Large differences may indicate some neighborhoods have more 
              challenging conditions (higher demand or less solar).
            </p>
          </div>
        </div>
      )
    },
    metrics: {
      title: 'Key Metrics',
      content: (
        <div className="space-y-4">
          <div className="space-y-3">
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-amber-500"></div>
                <span className="text-amber-400 font-medium text-sm">Grid Imports (kWh)</span>
              </div>
              <p className="text-slate-400 text-xs leading-relaxed">
                Energy purchased from the main power grid. Lower is better - means more reliance on 
                local solar generation and efficient trading between neighborhoods.
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                <span className="text-emerald-400 font-medium text-sm">Solar Utilization (%)</span>
              </div>
              <p className="text-slate-400 text-xs leading-relaxed">
                Percentage of generated solar energy actually used. Higher is better - means less 
                wasted renewable energy. Can exceed 100% if stored solar is used later.
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-cyan-500"></div>
                <span className="text-cyan-400 font-medium text-sm">Demand Satisfaction (%)</span>
              </div>
              <p className="text-slate-400 text-xs leading-relaxed">
                Percentage of time steps where neighborhoods had enough energy to meet demand. 
                Higher is better - 100% means no blackouts occurred.
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-violet-500"></div>
                <span className="text-violet-400 font-medium text-sm">Battery Level (kWh)</span>
              </div>
              <p className="text-slate-400 text-xs leading-relaxed">
                Current energy stored in each neighborhood&apos;s battery. Ideal range is 40-60% of 
                capacity (40-60 kWh) - enough buffer for emergencies but room to store surplus.
              </p>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 rounded-full bg-rose-500"></div>
                <span className="text-rose-400 font-medium text-sm">Energy Traded (kWh)</span>
              </div>
              <p className="text-slate-400 text-xs leading-relaxed">
                Energy exchanged between neighborhoods. High trading indicates active cooperation 
                and efficient resource sharing across the grid.
              </p>
            </div>
          </div>
        </div>
      )
    },
    agents: {
      title: 'How Agents Learn',
      content: (
        <div className="space-y-4">
          <p className="text-slate-300 text-sm leading-relaxed">
            Each neighborhood is controlled by an AI agent trained using <span className="text-cyan-400 font-medium">Proximal Policy Optimization (PPO)</span>, 
            a state-of-the-art reinforcement learning algorithm.
          </p>

          <div className="space-y-2">
            <h4 className="text-amber-400 font-semibold text-sm">Agent Decision Process</h4>
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="space-y-2 text-xs text-slate-400">
                <div className="flex items-start gap-2">
                  <span className="text-cyan-400 font-bold">1.</span>
                  <span><strong className="text-slate-200">Observe</strong> - Current battery level, demand, solar generation, neighbor states</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-cyan-400 font-bold">2.</span>
                  <span><strong className="text-slate-200">Decide</strong> - How much to use grid power vs battery, how much to trade</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-cyan-400 font-bold">3.</span>
                  <span><strong className="text-slate-200">Act</strong> - Execute grid trading and neighbor energy transfers</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-cyan-400 font-bold">4.</span>
                  <span><strong className="text-slate-200">Learn</strong> - Receive reward, update policy to improve future decisions</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <h4 className="text-slate-200 font-medium text-sm mb-2">Multi-Agent Coordination</h4>
            <p className="text-slate-400 text-xs leading-relaxed">
              Unlike single-agent RL, our agents must learn to cooperate while competing for limited resources. 
              The reward function encourages sharing energy with struggling neighbors, creating emergent 
              cooperative behavior without explicit communication protocols.
            </p>
          </div>
        </div>
      )
    }
  };

  const sectionKeys = Object.keys(sections);

  return (
    <div className={`bg-slate-900/80 backdrop-blur-sm rounded-xl border border-slate-700/50 overflow-hidden ${className}`}>
      <div className="p-4 border-b border-slate-700/50">
        <h2 className="text-lg font-semibold text-slate-100">Understanding the Simulation</h2>
        <p className="text-slate-400 text-xs mt-1">Click a topic to learn more</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-slate-700/50 overflow-x-auto">
        {sectionKeys.map((key) => (
          <button
            key={key}
            onClick={() => setActiveSection(activeSection === key ? null : key)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors whitespace-nowrap
              ${activeSection === key 
                ? 'text-cyan-400 bg-slate-800/50 border-b-2 border-cyan-400' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/30'
              }`}
          >
            {sections[key].title}
          </button>
        ))}
      </div>

      {/* Content Panel */}
      <AnimatePresence mode="wait">
        {activeSection && (
          <motion.div
            key={activeSection}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="p-4"
          >
            {sections[activeSection].content}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

