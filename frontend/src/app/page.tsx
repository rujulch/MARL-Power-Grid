'use client';

import React, { useState, useEffect } from 'react';
import GridVisualization from '@/components/GridVisualization';
import AgentCard from '@/components/AgentCard';
import MetricsDashboard from '@/components/MetricsDashboard';
import ControlPanel from '@/components/ControlPanel';
import ObjectivesPanel from '@/components/ObjectivesPanel';
import DailyComparisonTable from '@/components/DailyComparisonTable';
import InfoPanel from '@/components/InfoPanel';
import ComparisonDashboard from '@/components/ComparisonDashboard';
import TrainingCurves from '@/components/TrainingCurves';
import ExportButtons from '@/components/ExportButtons';
import ScalabilityChart from '@/components/ScalabilityChart';
import { GridWebSocket } from '@/lib/websocket';
import { gridApi } from '@/lib/api';
import { GridState, DailySummary, SimulationCompleteData } from '@/types/grid';
import { Activity, Zap } from 'lucide-react';
import { getNeighborhoodDisplayName } from '@/lib/utils';

export default function Home() {
  const [gridState, setGridState] = useState<GridState | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [simulationComplete, setSimulationComplete] = useState(false);
  const [ws, setWs] = useState<GridWebSocket | null>(null);
  
  // Multi-day tracking
  const [dailySummaries, setDailySummaries] = useState<DailySummary[]>([]);
  const [overallMetrics, setOverallMetrics] = useState<SimulationCompleteData['overall_metrics'] | null>(null);
  const [completedDays, setCompletedDays] = useState(0);

  // Initialize WebSocket connection
  useEffect(() => {
    const websocket = new GridWebSocket('ws://localhost:8000/ws/grid');
    
    websocket.onConnect(() => {
      console.log('Connected to Grid API');
      setIsConnected(true);
    });

    websocket.onMessage((data: any) => {
      if (data.type === 'simulation_complete') {
        setIsSimulationRunning(false);
        setSimulationComplete(true);
        // Store final summaries
        if (data.daily_summaries) {
          setDailySummaries(data.daily_summaries);
        }
        if (data.overall_metrics) {
          setOverallMetrics(data.overall_metrics);
        }
      } else if (data.type === 'day_complete') {
        // Add daily summary as each day completes
        setDailySummaries(prev => {
          const existing = prev.find(s => s.day === data.summary.day);
          if (existing) return prev;
          return [...prev, data.summary];
        });
        setCompletedDays(data.day);
      } else if (data.type === 'state_update' || data.agents) {
        setGridState(data as GridState);
        setSimulationComplete(false);
      }
    });

    websocket.onError((error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    });

    websocket.connect();
    setWs(websocket);

    // Cleanup on unmount
    return () => {
      websocket.disconnect();
    };
  }, []);

  const handleStartSimulation = async () => {
    try {
      setSimulationComplete(false);
      setDailySummaries([]);
      setOverallMetrics(null);
      setCompletedDays(0);
      await gridApi.startSimulation({
        num_days: 7,
        steps_per_day: 288,
        scenario: 'default'
      });
      setIsSimulationRunning(true);
    } catch (error) {
      console.error('Error starting simulation:', error);
      alert('Failed to start simulation. Make sure the backend is running.');
    }
  };

  const handleStopSimulation = async () => {
    try {
      await gridApi.stopSimulation();
      setIsSimulationRunning(false);
    } catch (error) {
      console.error('Error stopping simulation:', error);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
      {/* Header */}
      <header className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <div className="p-2 bg-cyan-500/20 rounded-lg">
                <Zap className="w-7 h-7 text-cyan-400" />
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
                Smart Grid MARL
              </h1>
            </div>
            <p className="text-slate-400 text-sm ml-14">
              Multi-Agent Reinforcement Learning for Distributed Energy Optimization
            </p>
          </div>
          
          {/* Control Panel */}
          <ControlPanel
            onStart={handleStartSimulation}
            onStop={handleStopSimulation}
            isRunning={isSimulationRunning}
            isConnected={isConnected}
          />
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* Left Column - Objectives Panel */}
        <div className="xl:col-span-3">
          <ObjectivesPanel
            simulationTime={gridState?.simulation_time}
            progress={gridState?.progress ?? 0}
            dayProgress={gridState?.day_progress ?? 0}
            step={gridState?.step ?? 0}
            totalSteps={gridState?.total_steps ?? 2016}
            currentDay={gridState?.current_day ?? 1}
            totalDays={gridState?.total_days ?? 7}
            usingTrainedModel={gridState?.using_trained_model ?? null}
            metrics={gridState?.metrics}
            completedDays={completedDays}
            stability={gridState?.stability ?? 0}
          />
        </div>

        {/* Center - Main Grid Visualization */}
        <div className="xl:col-span-6">
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-4 h-full">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-slate-200">
                Energy Grid Network
              </h2>
              {gridState?.simulation_time && (
                <div className="flex items-center gap-4">
                  <span className="text-sm text-slate-400">
                    Grid Stability: 
                    <span className={`ml-2 font-mono ${
                      gridState.stability > 0.8 ? 'text-emerald-400' : 
                      gridState.stability > 0.6 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                      {(gridState.stability * 100).toFixed(1)}%
                    </span>
                  </span>
                </div>
              )}
            </div>
            <div className="h-[520px]">
              {gridState ? (
                <GridVisualization
                  agents={gridState.agents}
                  flows={gridState.energy_flows}
                  stability={gridState.stability}
                />
              ) : (
                <div className="flex items-center justify-center h-full bg-slate-900/50 rounded-xl border border-slate-700/50">
                  <div className="text-center">
                    <Activity className="w-16 h-16 text-slate-600 mx-auto mb-4 animate-pulse" />
                    <p className="text-slate-400 text-lg font-medium">
                      {isConnected 
                        ? 'Click "Start Simulation" to begin'
                        : 'Connecting to backend...'
                      }
                    </p>
                    <p className="text-slate-500 text-sm mt-2">
                      {isConnected && 'Watch trained AI agents optimize the energy grid'}
                    </p>
                  </div>
                </div>
              )}
            </div>
            
            {/* Simulation Complete Banner */}
            {simulationComplete && (
              <div className="mt-4 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-emerald-500/20 rounded-full flex items-center justify-center">
                    <Activity className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <h3 className="text-emerald-400 font-semibold">Simulation Complete!</h3>
                    <p className="text-slate-400 text-sm">
                      7 days simulated. See comparison table below. Click Start to run again.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column - Agent Cards + Metrics */}
        <div className="xl:col-span-3 space-y-5">
          {/* Metrics Dashboard */}
          {gridState && (
            <MetricsDashboard
              metrics={gridState.metrics}
              stability={gridState.stability}
            />
          )}
          
          {/* Agent Cards */}
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-4">
            <h2 className="text-lg font-bold text-slate-200 mb-3">
              Neighborhood Agents
            </h2>
            <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
              {gridState && gridState.agents.length > 0 ? (
                gridState.agents.map((agent) => (
                  <AgentCard
                    key={agent.id}
                    agent={{...agent, name: getNeighborhoodDisplayName(agent.id)}}
                    maxCapacity={100}
                  />
                ))
              ) : (
                <div className="text-center py-6">
                  <p className="text-slate-500 text-sm">
                    Waiting for simulation...
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Daily Comparison Table - shown when we have daily data */}
      {dailySummaries.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-bold text-slate-200">Simulation Results</h2>
            <ExportButtons 
              dailySummaries={dailySummaries}
              overallMetrics={overallMetrics ?? undefined}
            />
          </div>
          <DailyComparisonTable 
            dailySummaries={dailySummaries}
            overallMetrics={overallMetrics ?? undefined}
          />
        </div>
      )}

      {/* Policy Comparison Dashboard */}
      <div className="mt-6">
        <ComparisonDashboard />
      </div>

      {/* Training Progress Visualization */}
      <div className="mt-6">
        <TrainingCurves />
      </div>

      {/* Scalability Analysis */}
      <div className="mt-6">
        <ScalabilityChart />
      </div>

      {/* Info Panel - Educational Content */}
      <div className="mt-6">
        <InfoPanel />
      </div>

      {/* Info Footer */}
      <footer className="mt-6 flex items-center justify-between text-slate-500 text-xs">
        <p>
          Multi-Agent RL System • PPO Algorithm via Ray RLlib • PettingZoo Environment
        </p>
        <div className="flex items-center gap-4">
          <span className={`flex items-center gap-1.5 ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`} />
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
          <span className="text-slate-600">|</span>
          <span>
            {isSimulationRunning ? 'Running' : simulationComplete ? 'Complete' : 'Idle'}
          </span>
        </div>
      </footer>
    </main>
  );
}







