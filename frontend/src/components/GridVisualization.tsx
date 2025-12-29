'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { AgentState, EnergyFlow } from '@/types/grid';
import { getEnergyColor, getNeighborhoodProfile, NEIGHBORHOOD_PROFILES } from '@/lib/utils';

interface GridVisualizationProps {
  agents: AgentState[];
  flows: EnergyFlow[];
  stability: number;
}

// Base dimensions that agent positions are calculated for
const BASE_WIDTH = 1000;
const BASE_HEIGHT = 600;

export default function GridVisualization({ agents, flows, stability }: GridVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: BASE_WIDTH, height: BASE_HEIGHT });

  // Calculate scale factors
  const scaleX = dimensions.width / BASE_WIDTH;
  const scaleY = dimensions.height / BASE_HEIGHT;
  const scale = Math.min(scaleX, scaleY);

  // Handle responsive sizing with ResizeObserver
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setDimensions({ width, height });
        }
      }
    });

    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, []);

  // Scale a position from base coordinates to current dimensions
  const scalePosition = useCallback((x: number, y: number) => {
    return {
      x: (x / BASE_WIDTH) * dimensions.width,
      y: (y / BASE_HEIGHT) * dimensions.height
    };
  }, [dimensions]);

  useEffect(() => {
    if (!svgRef.current || agents.length === 0) return;

    const svg = d3.select(svgRef.current);
    const { width, height } = dimensions;

    // Clear previous render
    svg.selectAll('*').remove();

    // Create container group
    const g = svg.append('g');

    // Scaled node sizes
    const nodeRadius = Math.max(25, 45 * scale);
    const outerRadius = Math.max(30, 55 * scale);
    const arcInner = Math.max(20, 35 * scale);
    const arcOuter = Math.max(24, 42 * scale);
    const fontSize = Math.max(10, 14 * scale);
    const smallFontSize = Math.max(8, 12 * scale);

    // Draw energy flow lines
    const flowGroup = g.append('g').attr('class', 'flows');
    
    flows.forEach(flow => {
      const sourceAgent = agents.find(a => a.id === flow.source);
      const targetAgent = agents.find(a => a.id === flow.target);
      
      if (sourceAgent && targetAgent) {
        const source = scalePosition(sourceAgent.x, sourceAgent.y);
        const target = scalePosition(targetAgent.x, targetAgent.y);
        
        const line = flowGroup
          .append('line')
          .attr('x1', source.x)
          .attr('y1', source.y)
          .attr('x2', target.x)
          .attr('y2', target.y)
          .attr('stroke', flow.amount > 0 ? '#10b981' : '#3b82f6')
          .attr('stroke-width', Math.max(1, Math.abs(flow.amount) * 5 * scale))
          .attr('stroke-opacity', 0.4)
          .attr('stroke-dasharray', `${10 * scale},${5 * scale}`)
          .attr('class', 'energy-flow-line');

        // Animate flow
        line
          .attr('stroke-dashoffset', 1000)
          .transition()
          .duration(2000)
          .ease(d3.easeLinear)
          .attr('stroke-dashoffset', 0)
          .on('end', function repeat() {
            d3.select(this)
              .attr('stroke-dashoffset', 1000)
              .transition()
              .duration(2000)
              .ease(d3.easeLinear)
              .attr('stroke-dashoffset', 0)
              .on('end', repeat);
          });
      }
    });

    // Draw agent nodes
    const nodeGroup = g.append('g').attr('class', 'nodes');

    agents.forEach(agent => {
      const pos = scalePosition(agent.x, agent.y);
      
      const nodeG = nodeGroup.append('g')
        .attr('transform', `translate(${pos.x}, ${pos.y})`);

      // Outer glow
      nodeG.append('circle')
        .attr('r', outerRadius)
        .attr('fill', 'none')
        .attr('stroke', getEnergyColor(agent.energy_level, 100))
        .attr('stroke-width', Math.max(1, 2 * scale))
        .attr('stroke-opacity', 0.3)
        .attr('class', 'animate-pulse-slow');

      // Main circle
      nodeG.append('circle')
        .attr('r', nodeRadius)
        .attr('fill', getEnergyColor(agent.energy_level, 100))
        .attr('fill-opacity', 0.8)
        .attr('stroke', '#1f2937')
        .attr('stroke-width', Math.max(2, 3 * scale))
        .style('cursor', 'pointer')
        .on('mouseenter', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', nodeRadius * 1.1);
        })
        .on('mouseleave', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', nodeRadius);
        });

      // Energy level arc
      const arcGenerator = d3.arc()
        .innerRadius(arcInner)
        .outerRadius(arcOuter)
        .startAngle(0)
        .endAngle((agent.energy_level / 100) * 2 * Math.PI);

      nodeG.append('path')
        .attr('d', arcGenerator as any)
        .attr('fill', '#ffffff')
        .attr('fill-opacity', 0.3);

      // Agent label - show neighborhood type icon
      const profile = getNeighborhoodProfile(agent.id);
      nodeG.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', -fontSize * 0.3)
        .attr('fill', '#ffffff')
        .attr('font-size', `${fontSize}px`)
        .attr('font-weight', 'bold')
        .text(profile?.icon || agent.id.replace('agent_', 'A'));

      // Energy value
      nodeG.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', fontSize * 0.8)
        .attr('fill', '#ffffff')
        .attr('font-size', `${smallFontSize}px`)
        .text(`${agent.energy_level.toFixed(0)}kW`);

      // Generation indicator (sun icon)
      if (agent.generation > 0.5) {
        nodeG.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', -outerRadius - 10 * scale)
          .attr('font-size', `${Math.max(14, 20 * scale)}px`)
          .text('☀️');
      }

      // Demand indicator (red dot)
      if (agent.demand > 8) {
        nodeG.append('circle')
          .attr('cx', nodeRadius * 0.7)
          .attr('cy', -nodeRadius * 0.7)
          .attr('r', Math.max(4, 8 * scale))
          .attr('fill', '#ef4444')
          .attr('class', 'animate-pulse');
      }
    });

    // Add grid stability indicator (positioned relative to container)
    const stabilityWidth = Math.max(80, 120 * scale);
    const stabilityGroup = svg.append('g')
      .attr('transform', `translate(${width - stabilityWidth - 20}, 20)`);

    stabilityGroup.append('text')
      .attr('x', 0)
      .attr('y', 0)
      .attr('fill', '#94a3b8')
      .attr('font-size', `${Math.max(10, 14 * scale)}px`)
      .text('Grid Stability');

    stabilityGroup.append('rect')
      .attr('x', 0)
      .attr('y', 10)
      .attr('width', stabilityWidth)
      .attr('height', Math.max(14, 20 * scale))
      .attr('fill', '#1e293b')
      .attr('stroke', '#475569')
      .attr('rx', Math.max(6, 10 * scale));

    stabilityGroup.append('rect')
      .attr('x', 2)
      .attr('y', 12)
      .attr('width', Math.max(0, stability * (stabilityWidth - 4)))
      .attr('height', Math.max(10, 16 * scale))
      .attr('fill', getEnergyColor(stability * 100, 100))
      .attr('rx', Math.max(4, 8 * scale));

    stabilityGroup.append('text')
      .attr('x', stabilityWidth / 2)
      .attr('y', 10 + Math.max(14, 20 * scale) / 2 + 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ffffff')
      .attr('font-size', `${Math.max(9, 12 * scale)}px`)
      .attr('font-weight', 'bold')
      .text(`${(stability * 100).toFixed(0)}%`);

  }, [agents, flows, stability, dimensions, scale, scalePosition]);

  return (
    <div 
      ref={containerRef}
      className="w-full h-full bg-slate-900 rounded-xl border border-slate-700 overflow-hidden"
    >
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="w-full h-full"
      />
    </div>
  );
}







