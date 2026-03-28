'use client';

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { ChartSpline, ChartScatter, Trash2, Download } from 'lucide-react';
import { useBandStructureWasm, StreamingBandData } from './useBandStructureWasm';

type LatticeType = 'square' | 'triangular' | 'rectangular' | 'honeycomb';

// Get color based on epsilon using greyscale (high epsilon = dark, low epsilon = white)
function getEpsilonColor(epsilon: number): string {
  const normalized = (epsilon - 1) / 12;
  const grey = Math.round(180 - normalized * 110);
  return `rgb(${grey}, ${grey}, ${grey})`;
}

interface LatticeConfig {
  name: string;
  getBasisVectors: (param?: number) => { a1: [number, number]; a2: [number, number] };
  atomPositions: [number, number][];
  hasParameter?: boolean;
  parameterRange?: [number, number];
  parameterName?: string;
  pathPreset: string;
  pathLabels: string[];
  /** Total k-path distance in fractional coordinates */
  pathTotalDistance: number;
}

// Calculate path distances in fractional k-space
// Square: Γ(0,0) → X(0.5,0) → M(0.5,0.5) → Γ(0,0)
//   Γ→X: 0.5, X→M: 0.5, M→Γ: √0.5 ≈ 0.707
//   Total: 0.5 + 0.5 + 0.707 ≈ 1.707
// Triangular: Γ(0,0) → M(0.5,0) → K(1/3,1/3) → Γ(0,0)
//   Γ→M: 0.5, M→K: √((0.5-1/3)² + (1/3)²) ≈ 0.373, K→Γ: √(2/9) ≈ 0.471
//   Total: 0.5 + 0.373 + 0.471 ≈ 1.344
// Rectangular: Γ(0,0) → X(0.5,0) → S(0.5,0.5) → Y(0,0.5) → Γ(0,0)
//   Γ→X: 0.5, X→S: 0.5, S→Y: 0.5, Y→Γ: 0.5
//   Total: 2.0

const BRAVAIS_LATTICES: Record<Exclude<LatticeType, 'honeycomb'>, LatticeConfig> = {
  square: {
    name: 'Square',
    getBasisVectors: () => ({
      a1: [1, 0],
      a2: [0, 1],
    }),
    atomPositions: [[0, 0]],
    pathPreset: 'square',
    pathLabels: ['Γ', 'X', 'M', 'Γ'],
    pathTotalDistance: 0.5 + 0.5 + Math.sqrt(0.5), // ≈ 1.707
  },
  triangular: {
    name: 'Triangular',
    getBasisVectors: () => ({
      a1: [1, 0],
      a2: [0.5, Math.sqrt(3) / 2],
    }),
    atomPositions: [[0, 0]],
    pathPreset: 'triangular',
    pathLabels: ['Γ', 'M', 'K', 'Γ'],
    pathTotalDistance: 0.5 + Math.sqrt(Math.pow(0.5 - 1/3, 2) + Math.pow(1/3, 2)) + Math.sqrt(2/9), // ≈ 1.344
  },
  rectangular: {
    name: 'Rectangular',
    getBasisVectors: (b = 1) => ({
      a1: [1, 0],
      a2: [0, b],
    }),
    atomPositions: [[0, 0]],
    hasParameter: true,
    parameterRange: [0.5, 2],
    parameterName: 'b',
    pathPreset: 'rectangular',
    pathLabels: ['Γ', 'X', 'S', 'Y', 'Γ'],
    pathTotalDistance: 2.0, // 0.5 + 0.5 + 0.5 + 0.5
  },
};

const SPECIAL_LATTICES: Record<'honeycomb', LatticeConfig> = {
  honeycomb: {
    name: 'Honeycomb',
    getBasisVectors: () => ({
      a1: [1, 0],
      a2: [0.5, Math.sqrt(3) / 2],
    }),
    atomPositions: [[0, 0], [1/3, 1/3]],
    pathPreset: 'triangular',
    pathLabels: ['Γ', 'M', 'K', 'Γ'],
    pathTotalDistance: 0.5 + Math.sqrt(Math.pow(0.5 - 1/3, 2) + Math.pow(1/3, 2)) + Math.sqrt(2/9), // ≈ 1.344
  },
};

const ALL_LATTICES = { ...BRAVAIS_LATTICES, ...SPECIAL_LATTICES };

// =============================================================================
// Compact Crystal Builder Component
// =============================================================================

interface CompactCrystalBuilderProps {
  latticeType: LatticeType;
  setLatticeType: (type: LatticeType) => void;
  radius: number;
  setRadius: (r: number) => void;
  rectangularB: number;
  setRectangularB: (b: number) => void;
  backgroundEpsilon: number;
  setBackgroundEpsilon: (e: number) => void;
  circleEpsilon: number;
  setCircleEpsilon: (e: number) => void;
  isComputing: boolean;
}

function CompactCrystalBuilder({
  latticeType,
  setLatticeType,
  radius,
  setRadius,
  rectangularB,
  setRectangularB,
  backgroundEpsilon,
  setBackgroundEpsilon,
  circleEpsilon,
  setCircleEpsilon,
  isComputing,
}: CompactCrystalBuilderProps) {
  const lattice = ALL_LATTICES[latticeType];
  const basisVectors = useMemo(
    () => lattice.getBasisVectors(rectangularB),
    [lattice, rectangularB]
  );

  // Generate lattice points for preview
  const latticePoints = useMemo(() => {
    const points: { x: number; y: number }[] = [];
    const { a1, a2 } = basisVectors;
    
    // More shells for triangular/honeycomb/rectangular
    const range = (latticeType === 'triangular' || latticeType === 'honeycomb' || latticeType === 'rectangular') ? 4 : 3;
    for (let i = -range; i <= range; i++) {
      for (let j = -range; j <= range; j++) {
        for (const [fx, fy] of lattice.atomPositions) {
          const x = (i + fx) * a1[0] + (j + fy) * a2[0];
          const y = (i + fx) * a1[1] + (j + fy) * a2[1];
          points.push({ x, y });
        }
      }
    }
    return points;
  }, [basisVectors, lattice.atomPositions, latticeType]);

  const viewBounds = useMemo(() => {
    const { a1, a2 } = basisVectors;
    const maxExtent = Math.max(
      Math.abs(a1[0]) + Math.abs(a2[0]),
      Math.abs(a1[1]) + Math.abs(a2[1])
    );
    // Zoom in for triangular/honeycomb (smaller multiplier = more zoomed in)
    const zoomMultiplier = (latticeType === 'triangular' || latticeType === 'honeycomb') ? 1.5 : 2;
    return { min: -maxExtent * zoomMultiplier, max: maxExtent * zoomMultiplier, size: maxExtent * zoomMultiplier * 2 };
  }, [basisVectors, latticeType]);

  const bgColor = getEpsilonColor(backgroundEpsilon);
  const circleColor = getEpsilonColor(circleEpsilon);

  const buttonStyle = (isSelected: boolean) => ({
    padding: '0.4rem 0.6rem',
    borderRadius: '6px',
    border: '2px solid ' + (isSelected ? 'rgba(100, 200, 255, 0.8)' : 'transparent'),
    background: isSelected 
      ? 'rgba(100, 200, 255, 0.2)' 
      : 'rgba(255,255,255,0.05)',
    color: 'white',
    cursor: isComputing ? 'not-allowed' : 'pointer',
    fontSize: '0.75rem',
    flex: 1,
    opacity: isComputing ? 0.5 : 1,
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Preview - 50% smaller */}
      <div style={{
        background: bgColor,
        borderRadius: '4px',
        overflow: 'hidden',
        aspectRatio: '1',
        width: '100%',
        maxWidth: '200px',
        margin: '0 auto',
        position: 'relative',
      }}>
        <svg
          viewBox={`${viewBounds.min} ${viewBounds.min} ${viewBounds.size} ${viewBounds.size}`}
          style={{ width: '100%', height: '100%' }}
        >
          {latticePoints.map((point, i) => (
            <circle
              key={i}
              cx={point.x}
              cy={-point.y}
              r={radius}
              fill={circleColor}
            />
          ))}
          <g stroke="rgba(48, 105, 142, 0.5)" strokeWidth="0.03" fill="none" strokeDasharray="0.1">
            <path d={`M 0 0 L ${basisVectors.a1[0]} ${-basisVectors.a1[1]} L ${basisVectors.a1[0] + basisVectors.a2[0]} ${-(basisVectors.a1[1] + basisVectors.a2[1])} L ${basisVectors.a2[0]} ${-basisVectors.a2[1]} Z`} />
          </g>
          <defs>
            <marker id="arrowhead-compact" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
              <polygon points="0 0, 6 2, 0 4" fill="#30698e" />
            </marker>
          </defs>
          <line
            x1="0" y1="0"
            x2={basisVectors.a1[0]} y2={-basisVectors.a1[1]}
            stroke="#30698e"
            strokeWidth="0.04"
            markerEnd="url(#arrowhead-compact)"
          />
          <line
            x1="0" y1="0"
            x2={basisVectors.a2[0]} y2={-basisVectors.a2[1]}
            stroke="#30698e"
            strokeWidth="0.04"
            markerEnd="url(#arrowhead-compact)"
          />
        </svg>
      </div>

      {/* Epsilon Color scale */}
      <div>
        <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.65rem', marginBottom: '0.25rem' }}>
          Epsilon Color Scale
        </div>
        <div style={{
          height: '12px',
          borderRadius: '3px',
          background: 'linear-gradient(to right, rgb(255, 255, 255) 0%, rgb(40, 40, 40) 100%)',
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'rgba(255,255,255,0.4)', marginTop: '0.15rem' }}>
          <span>ε = 1</span>
          <span>ε = 13</span>
        </div>
      </div>

      {/* Lattice Selection */}
      <div>
        <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.75rem', marginBottom: '0.25rem', display: 'block' }}>
          Bravais Lattice
        </label>
        <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
          {(Object.keys(BRAVAIS_LATTICES) as Exclude<LatticeType, 'honeycomb'>[]).map((type) => (
            <button
              key={type}
              onClick={() => setLatticeType(type)}
              disabled={isComputing}
              style={buttonStyle(latticeType === type)}
            >
              {BRAVAIS_LATTICES[type].name}
            </button>
          ))}
        </div>
      </div>

      {/* Special Lattices */}
      <div>
        <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.75rem', marginBottom: '0.25rem', display: 'block' }}>
          Special
        </label>
        <div style={{ display: 'flex', gap: '0.25rem' }}>
          {(Object.keys(SPECIAL_LATTICES) as ('honeycomb')[]).map((type) => (
            <button
              key={type}
              onClick={() => setLatticeType(type)}
              disabled={isComputing}
              style={buttonStyle(latticeType === type)}
            >
              {SPECIAL_LATTICES[type].name}
            </button>
          ))}
        </div>
      </div>

      {/* Rectangular parameter */}
      {latticeType === 'rectangular' && (
        <div style={{ opacity: isComputing ? 0.5 : 1 }}>
          <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.75rem' }}>
            b: {rectangularB.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.5"
            max="2"
            step="0.01"
            value={rectangularB}
            onChange={(e) => setRectangularB(parseFloat(e.target.value))}
            disabled={isComputing}
            style={{ width: '100%', accentColor: 'rgba(100, 200, 255, 0.8)' }}
          />
        </div>
      )}

      {/* Sliders */}
      <div style={{ opacity: isComputing ? 0.5 : 1 }}>
        <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.75rem' }}>
          Radius: {radius.toFixed(2)}
        </label>
        <input
          type="range"
          min="0.05"
          max="0.45"
          step="0.01"
          value={radius}
          onChange={(e) => setRadius(parseFloat(e.target.value))}
          disabled={isComputing}
          style={{ width: '100%', accentColor: 'rgba(100, 200, 255, 0.8)' }}
        />
      </div>

      <div style={{ opacity: isComputing ? 0.5 : 1 }}>
        <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.75rem' }}>
          Background ε: {backgroundEpsilon.toFixed(1)}
        </label>
        <input
          type="range"
          min="1"
          max="13"
          step="0.1"
          value={backgroundEpsilon}
          onChange={(e) => setBackgroundEpsilon(parseFloat(e.target.value))}
          disabled={isComputing}
          style={{ width: '100%', accentColor: 'rgba(100, 200, 255, 0.8)' }}
        />
      </div>

      <div style={{ opacity: isComputing ? 0.5 : 1 }}>
        <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.75rem' }}>
          Circle ε: {circleEpsilon.toFixed(1)}
        </label>
        <input
          type="range"
          min="1"
          max="13"
          step="0.1"
          value={circleEpsilon}
          onChange={(e) => setCircleEpsilon(parseFloat(e.target.value))}
          disabled={isComputing}
          style={{ width: '100%', accentColor: 'rgba(100, 200, 255, 0.8)' }}
        />
      </div>
    </div>
  );
}

// =============================================================================
// Live Band Plot Component
// =============================================================================

interface LiveBandPlotProps {
  tmData: StreamingBandData | null;
  teData: StreamingBandData | null;
  phase: 'idle' | 'tm' | 'te' | 'done';
  pathLabels: string[];
  totalKPoints: number;
  pathTotalDistance: number;
  showDots: boolean;
  isStale: boolean;
  onToggleDots: () => void;
  onClear: () => void;
  onSave: () => void;
}

function LiveBandPlot({ tmData, teData, phase, pathLabels, totalKPoints, pathTotalDistance, showDots, isStale, onToggleDots, onClear, onSave }: LiveBandPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // For streaming mode, we just track the last data length to know when to redraw
  const lastKCountRef = useRef<{ tm: number; te: number }>({ tm: 0, te: 0 });
  
  // Lock y-axis after first data arrives (with +20% buffer)
  const lockedYMaxRef = useRef<number | null>(null);
  
  // Track data changes for redraw
  useEffect(() => {
    const tmKCount = tmData?.numKPoints ?? 0;
    const teKCount = teData?.numKPoints ?? 0;
    lastKCountRef.current = { tm: tmKCount, te: teKCount };
  }, [tmData, teData]);
  
  // Reset locked yMax when data is cleared (both null means fresh start)
  useEffect(() => {
    if (!tmData && !teData) {
      lockedYMaxRef.current = null;
    }
  }, [tmData, teData]);

  // Draw the plot - redraws whenever data changes
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw plot background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(padding.left, padding.top, plotWidth, plotHeight);

    // Draw axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    // Y axis label
    ctx.save();
    ctx.translate(15, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('ωa/2πc', 0, 0);
    ctx.restore();

    // X axis label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Wave vector k', padding.left + plotWidth / 2, height - 5);

    // High symmetry points
    const numLegs = pathLabels.length - 1;
    pathLabels.forEach((label, i) => {
      const x = padding.left + (i / numLegs) * plotWidth;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, padding.top + plotHeight + 16);
      
      if (i > 0 && i < pathLabels.length - 1) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotHeight);
        ctx.stroke();
      }
    });

    // Calculate y axis scale from data (bands is [kIndex][bandIndex])
    // Lock the y-axis after first k-point with +20% buffer
    let yMax = 0.8; // Default before any data
    
    if (lockedYMaxRef.current !== null) {
      // Use locked value
      yMax = lockedYMaxRef.current;
    } else {
      // Check if we have first data point to lock
      if (tmData && tmData.bands.length > 0) {
        const tmMax = Math.max(...tmData.bands.flatMap(kBands => kBands));
        // Lock with +20% buffer, rounded up to nearest 0.1 (no minimum)
        lockedYMaxRef.current = Math.ceil((tmMax * 1.2) * 10) / 10;
        yMax = lockedYMaxRef.current;
      }
    }

    // Y axis ticks
    const nYTicks = 4;
    for (let i = 0; i <= nYTicks; i++) {
      const val = (i / nYTicks) * yMax;
      const y = padding.top + plotHeight - (i / nYTicks) * plotHeight;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(2), padding.left - 6, y + 3);
      
      if (i > 0) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + plotWidth, y);
        ctx.stroke();
      }
    }

    // Helper to draw bands
    // data.bands is [kIndex][bandIndex], so we need to draw each band by iterating through k-points
    const drawBands = (
      data: StreamingBandData,
      color: string,
    ) => {
      if (data.bands.length === 0) return;

      const nBands = data.numBands;
      
      // Use the known total path distance in fractional k-space
      // This is pre-calculated for each lattice type
      const maxDist = pathTotalDistance;
      
      // Draw each band
      for (let bandIdx = 0; bandIdx < nBands; bandIdx++) {
        const opacity = 0.9 - (bandIdx / Math.max(nBands - 1, 1)) * 0.5;
        const bandColor = color.replace('1)', `${opacity})`);
        ctx.strokeStyle = bandColor;
        ctx.fillStyle = bandColor;
        ctx.lineWidth = 1.5;
        
        // Collect all valid points for this band
        const points: { x: number; y: number }[] = [];
        for (let kIdx = 0; kIdx < data.bands.length; kIdx++) {
          const kPointBands = data.bands[kIdx];
          if (!kPointBands || kPointBands[bandIdx] === undefined) continue;
          
          const x = padding.left + (data.distances[kIdx] / maxDist) * plotWidth;
          const y = padding.top + plotHeight - (kPointBands[bandIdx] / yMax) * plotHeight;
          points.push({ x, y });
        }
        
        // Draw dots at each k-point (if enabled)
        if (showDots) {
          for (const pt of points) {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        
        // Draw lines connecting points (if more than 1)
        if (points.length > 1) {
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
          }
          ctx.stroke();
        }
      }
    };

    // Draw TM bands (blue) - draw all available data
    if (tmData && tmData.bands.length > 0) {
      drawBands(tmData, 'rgba(100, 180, 255, 1)');
    }

    // Draw TE bands (orange) - draw all available data  
    if (teData && teData.bands.length > 0) {
      drawBands(teData, 'rgba(255, 160, 80, 1)');
    }

    // Draw legend
    if (tmData || teData) {
      const legendX = padding.left + plotWidth - 80;
      const legendY = padding.top + 20;
      
      if (tmData) {
        ctx.fillStyle = 'rgba(100, 180, 255, 0.9)';
        ctx.fillRect(legendX, legendY, 24, 6);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = 'bold 16px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('TM', legendX + 30, legendY + 7);
      }
      
      if (teData) {
        ctx.fillStyle = 'rgba(255, 160, 80, 0.9)';
        ctx.fillRect(legendX, legendY + 24, 24, 6);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = 'bold 16px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('TE', legendX + 30, legendY + 31);
      }
    }

    // Show waiting message if no data
    if (!tmData && !teData && phase === 'idle') {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click "Compute Band Diagram" to start', width / 2, height / 2);
    }

  }, [tmData, teData, phase, pathLabels, pathTotalDistance, showDots]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: '200px' }}>
      {/* Icon buttons container - above the plot */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginBottom: '4px' }}>
        {/* Save button - only show when there's data */}
        {(tmData || teData) && (
          <button
            onClick={onSave}
            style={{
              background: 'transparent',
              border: 'none',
              padding: '4px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            title="Save as CSV"
            className="icon-button"
          >
            <Download size={22} color="rgba(200, 200, 200, 0.6)" />
          </button>
        )}
        {/* Clear button - only show when there's data */}
        {(tmData || teData) && (
          <button
            onClick={onClear}
            style={{
              background: 'transparent',
              border: 'none',
              padding: '4px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            title="Clear data"
            className="icon-button"
          >
            <Trash2 size={22} color="rgba(200, 200, 200, 0.6)" />
          </button>
        )}
        {/* Toggle dots button */}
        <button
          onClick={onToggleDots}
          style={{
            background: 'transparent',
            border: 'none',
            padding: '4px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          title={showDots ? 'Show lines only' : 'Show dots'}
          className="icon-button"
        >
          {showDots ? (
            <ChartScatter size={22} color="rgba(200, 200, 200, 0.6)" />
          ) : (
            <ChartSpline size={22} color="rgba(200, 200, 200, 0.6)" />
          )}
        </button>
      </div>
      <div style={{ position: 'relative', flex: 1 }}>
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            borderRadius: '12px',
            opacity: isStale ? 0.4 : 1,
            transition: 'opacity 0.3s',
          }}
        />
        {isStale && (tmData || teData) && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            color: 'rgba(255, 255, 255, 0.8)',
            fontSize: '0.85rem',
            textAlign: 'center',
            pointerEvents: 'none',
            padding: '0.5rem 1rem',
            background: 'rgba(0, 0, 0, 0.5)',
            borderRadius: '8px',
            whiteSpace: 'nowrap',
          }}>
            Lattice parameters changed — recompute band diagram
          </div>
        )}
      </div>
      <style jsx>{`
        .icon-button:hover svg {
          stroke: white !important;
        }
      `}</style>
    </div>
  );
}

// =============================================================================
// Main Interactive Component
// =============================================================================

export default function InteractiveBandDiagram() {
  // Crystal builder state
  const [latticeType, setLatticeType] = useState<LatticeType>('square');
  const [radius, setRadius] = useState(0.25);
  const [rectangularB, setRectangularB] = useState(1.5);
  const [backgroundEpsilon, setBackgroundEpsilon] = useState(12.9);
  const [circleEpsilon, setCircleEpsilon] = useState(1);
  
  // Band plot display options
  const [showDots, setShowDots] = useState(false);
  
  // Track if data is stale (settings changed after computation)
  const [isStale, setIsStale] = useState(false);
  
  // Track the settings that were used for the last computation
  const lastComputedSettingsRef = useRef<string | null>(null);

  // WASM hook
  const {
    tmData,
    teData,
    isComputing,
    phase,
    tmProgress,
    teProgress,
    currentKIndex,
    totalKPoints,
    error,
    computeTime,
    tmTime,
    teTime,
    compute,
    reset,
    isInitialized,
  } = useBandStructureWasm();

  const lattice = ALL_LATTICES[latticeType];
  const basisVectors = useMemo(
    () => lattice.getBasisVectors(rectangularB),
    [lattice, rectangularB]
  );
  
  // Current settings as a string for comparison
  const currentSettings = useMemo(() => 
    JSON.stringify({ latticeType, radius, rectangularB, backgroundEpsilon, circleEpsilon }),
    [latticeType, radius, rectangularB, backgroundEpsilon, circleEpsilon]
  );
  
  // Detect when settings change after data exists
  useEffect(() => {
    if ((tmData || teData) && lastComputedSettingsRef.current && lastComputedSettingsRef.current !== currentSettings) {
      setIsStale(true);
    }
  }, [currentSettings, tmData, teData]);
  
  // Clear data when lattice type changes
  const handleLatticeTypeChange = useCallback((newType: LatticeType) => {
    if (newType !== latticeType && (tmData || teData)) {
      reset();
      setIsStale(false);
      lastComputedSettingsRef.current = null;
    }
    setLatticeType(newType);
  }, [latticeType, tmData, teData, reset]);

  // Generate TOML config from current state
  const generateConfig = useCallback(() => {
    const { a1, a2 } = basisVectors;
    
    // Map lattice type to TOML lattice type
    let tomlLatticeType = 'square';
    if (latticeType === 'triangular' || latticeType === 'honeycomb') {
      tomlLatticeType = 'triangular';
    } else if (latticeType === 'rectangular') {
      tomlLatticeType = 'rectangular';
    }

    // Generate atoms section
    const atoms = lattice.atomPositions.map(([fx, fy], idx) => {
      // Convert fractional to absolute position within unit cell
      const posX = fx * a1[0] + fy * a2[0];
      const posY = fx * a1[1] + fy * a2[1];
      // Normalize to [0, 1] range for the solver (add 0.5 to center in cell)
      return `[[geometry.atoms]]
pos = [${(posX + 0.5).toFixed(4)}, ${(posY + 0.5).toFixed(4)}]
radius = ${radius.toFixed(4)}
eps_inside = ${circleEpsilon.toFixed(1)}`;
    }).join('\n\n');

    // Calculate lattice dimensions for rectangular
    const lx = Math.sqrt(a1[0] * a1[0] + a1[1] * a1[1]);
    const ly = Math.sqrt(a2[0] * a2[0] + a2[1] * a2[1]);

    const config = `polarization = "TM"

[bulk]

[solver]
type = "maxwell"

[geometry]
eps_bg = ${backgroundEpsilon.toFixed(1)}

[geometry.lattice]
type = "${tomlLatticeType}"
a = 1.0

${atoms}

[grid]
nx = 32
ny = 32
lx = ${lx.toFixed(4)}
ly = ${ly.toFixed(4)}

[path]
preset = "${lattice.pathPreset}"
segments_per_leg = 15

[eigensolver]
n_bands = 8
max_iter = 200
tol = 1e-6
`;
    return config;
  }, [latticeType, radius, backgroundEpsilon, circleEpsilon, basisVectors, lattice, rectangularB]);

  const handleCompute = useCallback(() => {
    const config = generateConfig();
    lastComputedSettingsRef.current = currentSettings;
    setIsStale(false);
    compute(config);
  }, [generateConfig, compute, currentSettings]);

  return (
    <div style={{
      background: 'rgba(0, 0, 0, 0.4)',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      borderRadius: '24px',
      padding: '1.5rem',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      maxWidth: '1100px',
      width: '100%',
    }}>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '250px 1fr',
        gap: '1.5rem',
      }}>
        {/* Left: Compact Crystal Builder */}
        <CompactCrystalBuilder
          latticeType={latticeType}
          setLatticeType={handleLatticeTypeChange}
          radius={radius}
          setRadius={setRadius}
          rectangularB={rectangularB}
          setRectangularB={setRectangularB}
          backgroundEpsilon={backgroundEpsilon}
          setBackgroundEpsilon={setBackgroundEpsilon}
          circleEpsilon={circleEpsilon}
          setCircleEpsilon={setCircleEpsilon}
          isComputing={isComputing}
        />

        {/* Right: Band Plot */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <LiveBandPlot
            tmData={tmData}
            teData={teData}
            phase={phase}
            pathLabels={lattice.pathLabels}
            totalKPoints={totalKPoints}
            pathTotalDistance={lattice.pathTotalDistance}
            showDots={showDots}
            isStale={isStale}
            onToggleDots={() => setShowDots(!showDots)}
            onClear={reset}
            onSave={() => {
              // Build CSV content
              if (!tmData && !teData) return;
              
              // Get the number of bands for each polarization
              const tmNumBands = tmData?.numBands ?? 0;
              const teNumBands = teData?.numBands ?? 0;
              
              // Build comment line with crystal properties
              const props = [
                `lattice_type=${latticeType}`,
                `radius=${radius.toFixed(4)}`,
                `eps_background=${backgroundEpsilon.toFixed(1)}`,
                `eps_circle=${circleEpsilon.toFixed(1)}`,
                `k_path=${lattice.pathLabels.join('-')}`,
              ];
              if (latticeType === 'rectangular') {
                props.push(`b=${rectangularB.toFixed(4)}`);
              }
              const commentLine = `# ${props.join(', ')}`;
              
              // Build header
              const headers = ['k_distance'];
              for (let i = 1; i <= tmNumBands; i++) headers.push(`tm${i}`);
              for (let i = 1; i <= teNumBands; i++) headers.push(`te${i}`);
              
              // Get all unique k-distances and build rows
              const allDistances = new Set<number>();
              tmData?.distances.forEach(d => allDistances.add(d));
              teData?.distances.forEach(d => allDistances.add(d));
              const sortedDistances = Array.from(allDistances).sort((a, b) => a - b);
              
              // Build data rows (comment first, then header)
              const rows: string[] = [commentLine, headers.join(',')];
              for (const dist of sortedDistances) {
                const row: (string | number)[] = [dist.toFixed(6)];
                
                // Find TM data for this distance
                const tmIdx = tmData?.distances.findIndex(d => Math.abs(d - dist) < 1e-9) ?? -1;
                for (let b = 0; b < tmNumBands; b++) {
                  if (tmIdx >= 0 && tmData?.bands[tmIdx]?.[b] !== undefined) {
                    row.push(tmData.bands[tmIdx][b].toFixed(6));
                  } else {
                    row.push('');
                  }
                }
                
                // Find TE data for this distance
                const teIdx = teData?.distances.findIndex(d => Math.abs(d - dist) < 1e-9) ?? -1;
                for (let b = 0; b < teNumBands; b++) {
                  if (teIdx >= 0 && teData?.bands[teIdx]?.[b] !== undefined) {
                    row.push(teData.bands[teIdx][b].toFixed(6));
                  } else {
                    row.push('');
                  }
                }
                
                rows.push(row.join(','));
              }
              
              // Download
              const csv = rows.join('\n');
              const blob = new Blob([csv], { type: 'text/csv' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = 'band_structure.csv';
              a.click();
              URL.revokeObjectURL(url);
            }}
          />

          {/* Compute Button and Status */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '9rem' }}>
            <button
              onClick={handleCompute}
              disabled={isComputing || !isInitialized}
              style={{
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                background: isComputing
                  ? 'rgba(100, 200, 255, 0.3)'
                  : 'linear-gradient(135deg, rgba(100, 200, 255, 0.8), rgba(0, 255, 136, 0.6))',
                color: 'white',
                cursor: isComputing || !isInitialized ? 'not-allowed' : 'pointer',
                fontSize: '0.9rem',
                fontWeight: 600,
                transition: 'all 0.2s',
                opacity: !isInitialized ? 0.5 : 1,
                width: '240px',
                textAlign: 'center',
                flexShrink: 0,
              }}
            >
              {!isInitialized ? 'Loading WASM...' : isComputing ? 'Computing...' : 'Compute Band Diagram'}
            </button>

            {/* Dual Progress Bars */}
            <div style={{ 
              fontSize: '0.75rem', 
              color: 'rgba(255, 255, 255, 0.6)',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
            }}>
              {/* TM Progress */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: '24px', color: 'rgba(100, 180, 255, 0.9)' }}>TM</span>
                <div style={{
                  width: '225px',
                  height: '5px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    width: `${tmProgress}%`,
                    height: '100%',
                    background: 'rgba(100, 180, 255, 0.8)',
                    transition: 'width 0.1s',
                  }} />
                </div>
                <span style={{ minWidth: '60px', color: tmTime !== null ? 'rgba(100, 180, 255, 0.7)' : 'inherit' }}>
                  {tmTime !== null ? `${tmTime.toFixed(0)}ms` : (phase === 'tm' && totalKPoints > 0) ? `${currentKIndex + 1}/${totalKPoints}` : ''}
                </span>
              </div>
              
              {/* TE Progress */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: '24px', color: 'rgba(255, 160, 80, 0.9)' }}>TE</span>
                <div style={{
                  width: '225px',
                  height: '5px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    width: `${teProgress}%`,
                    height: '100%',
                    background: 'rgba(255, 160, 80, 0.8)',
                    transition: 'width 0.1s',
                  }} />
                </div>
                <span style={{ minWidth: '60px', color: teTime !== null ? 'rgba(255, 160, 80, 0.7)' : 'inherit' }}>
                  {teTime !== null ? `${teTime.toFixed(0)}ms` : (phase === 'te' && totalKPoints > 0) ? `${currentKIndex + 1}/${totalKPoints}` : ''}
                </span>
              </div>
              
              {error && (
                <span style={{ color: 'rgba(255, 100, 100, 0.8)' }}>
                  ✗ {error}
                </span>
              )}
            </div>
          </div>


        </div>
      </div>
    </div>
  );
}
