'use client';

import React from 'react';
import { Play, Square, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ControlPanelProps {
  onStart: () => void;
  onStop: () => void;
  isRunning: boolean;
  isConnected: boolean;
}

export default function ControlPanel({ onStart, onStop, isRunning, isConnected }: ControlPanelProps) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={onStart}
        disabled={isRunning || !isConnected}
        className={cn(
          "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          isRunning || !isConnected
            ? "bg-slate-700 text-slate-400"
            : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20"
        )}
      >
        {isRunning ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Running...
          </>
        ) : (
          <>
            <Play className="w-4 h-4" />
            Start Simulation
          </>
        )}
      </button>

      <button
        onClick={onStop}
        disabled={!isRunning}
        className={cn(
          "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          !isRunning
            ? "bg-slate-800 text-slate-500 border border-slate-700"
            : "bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-500/20"
        )}
      >
        <Square className="w-4 h-4" />
        Stop
      </button>
    </div>
  );
}







