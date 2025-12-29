'use client';

import React, { useState } from 'react';
import { Download, FileJson, FileSpreadsheet, Loader2 } from 'lucide-react';
import { DailySummary } from '@/types/grid';

interface ExportButtonsProps {
  dailySummaries?: DailySummary[];
  overallMetrics?: {
    avg_daily_reward: number;
    avg_stability: number;
    avg_grid_imports: number;
    avg_solar_utilization: number;
    avg_demand_satisfaction: number;
  };
  apiUrl?: string;
}

export default function ExportButtons({ 
  dailySummaries = [], 
  overallMetrics,
  apiUrl = 'http://localhost:8000'
}: ExportButtonsProps) {
  const [exporting, setExporting] = useState<string | null>(null);

  const storeAndExport = async (format: 'csv' | 'json') => {
    setExporting(format);
    
    try {
      // First, store the results on the backend
      await fetch(`${apiUrl}/api/export/store-results`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          daily_summaries: dailySummaries,
          overall_metrics: overallMetrics
        })
      });

      // Then trigger the download
      const endpoint = format === 'csv' ? '/api/export/csv' : '/api/export/json';
      const response = await fetch(`${apiUrl}${endpoint}`);
      
      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }
      
      // Get the blob and create download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `smart_grid_results_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Export error:', error);
      alert('Failed to export. Make sure backend is running.');
    } finally {
      setExporting(null);
    }
  };

  const exportClientSideCSV = () => {
    if (dailySummaries.length === 0) {
      alert('No data to export. Run a simulation first.');
      return;
    }

    // Build CSV content
    const headers = [
      'Day', 'Total Reward', 'Avg Stability (%)', 'Grid Imports (kWh)',
      'Solar Utilization (%)', 'Demand Satisfaction (%)'
    ];
    
    const rows = dailySummaries.map(s => [
      s.day,
      s.total_reward.toFixed(2),
      (s.avg_stability * 100).toFixed(2),
      s.grid_imports.toFixed(2),
      s.solar_utilization.toFixed(2),
      s.demand_satisfaction.toFixed(2)
    ]);

    let csvContent = headers.join(',') + '\n';
    csvContent += rows.map(r => r.join(',')).join('\n');

    if (overallMetrics) {
      csvContent += '\n\nOverall Metrics\n';
      csvContent += `Avg Daily Reward,${overallMetrics.avg_daily_reward.toFixed(2)}\n`;
      csvContent += `Avg Stability (%),${(overallMetrics.avg_stability * 100).toFixed(2)}\n`;
      csvContent += `Avg Grid Imports (kWh),${overallMetrics.avg_grid_imports.toFixed(2)}\n`;
      csvContent += `Avg Solar Utilization (%),${overallMetrics.avg_solar_utilization.toFixed(2)}\n`;
      csvContent += `Avg Demand Satisfaction (%),${overallMetrics.avg_demand_satisfaction.toFixed(2)}\n`;
    }

    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `smart_grid_results_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportClientSideJSON = () => {
    if (dailySummaries.length === 0) {
      alert('No data to export. Run a simulation first.');
      return;
    }

    const exportData = {
      export_timestamp: new Date().toISOString(),
      config: {
        num_agents: 5,
        max_energy_capacity: 100,
        total_days: dailySummaries.length
      },
      daily_summaries: dailySummaries,
      overall_metrics: overallMetrics
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `smart_grid_results_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const hasData = dailySummaries.length > 0;

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 mr-2">Export:</span>
      
      <button
        onClick={exportClientSideCSV}
        disabled={!hasData || exporting !== null}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors
          ${hasData 
            ? 'bg-emerald-600 hover:bg-emerald-500 text-white' 
            : 'bg-slate-700 text-slate-500 cursor-not-allowed'
          }`}
        title={hasData ? 'Export as CSV' : 'Run simulation first'}
      >
        {exporting === 'csv' ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <FileSpreadsheet className="w-3.5 h-3.5" />
        )}
        CSV
      </button>
      
      <button
        onClick={exportClientSideJSON}
        disabled={!hasData || exporting !== null}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors
          ${hasData 
            ? 'bg-cyan-600 hover:bg-cyan-500 text-white' 
            : 'bg-slate-700 text-slate-500 cursor-not-allowed'
          }`}
        title={hasData ? 'Export as JSON' : 'Run simulation first'}
      >
        {exporting === 'json' ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <FileJson className="w-3.5 h-3.5" />
        )}
        JSON
      </button>
    </div>
  );
}

