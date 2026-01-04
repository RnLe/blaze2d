'use client';

import { useState, useMemo } from 'react';

type LatticeType = 'square' | 'triangular' | 'rectangular' | 'honeycomb';

// Get color based on epsilon using greyscale (high epsilon = dark, low epsilon = white)
function getEpsilonColor(epsilon: number): string {
  // Map epsilon 1-13 to greyscale (white to dark grey)
  const normalized = (epsilon - 1) / 12;
  // 255 = white (low epsilon), 40 = very dark grey (high epsilon)
  const grey = Math.round(180 - normalized * 110);
  return `rgb(${grey}, ${grey}, ${grey})`;
}

interface LatticeConfig {
  name: string;
  getBasisVectors: (param?: number) => { a1: [number, number]; a2: [number, number] };
  atomPositions: [number, number][]; // In fractional coordinates
  hasParameter?: boolean;
  parameterRange?: [number, number];
  parameterName?: string;
}

const BRAVAIS_LATTICES: Record<Exclude<LatticeType, 'honeycomb'>, LatticeConfig> = {
  square: {
    name: 'Square',
    getBasisVectors: () => ({
      a1: [1, 0],
      a2: [0, 1],
    }),
    atomPositions: [[0, 0]],
  },
  triangular: {
    name: 'Triangular',
    getBasisVectors: () => ({
      a1: [1, 0],
      a2: [0.5, Math.sqrt(3) / 2],
    }),
    atomPositions: [[0, 0]],
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
  },
};

const ALL_LATTICES = { ...BRAVAIS_LATTICES, ...SPECIAL_LATTICES };

export default function CrystalBuilder() {
  const [latticeType, setLatticeType] = useState<LatticeType>('square');
  const [radius, setRadius] = useState(0.2);
  const [rectangularB, setRectangularB] = useState(1.0);
  const [backgroundEpsilon, setBackgroundEpsilon] = useState(1.0);
  const [circleEpsilon, setCircleEpsilon] = useState(11.7);

  const lattice = ALL_LATTICES[latticeType];
  const basisVectors = useMemo(
    () => lattice.getBasisVectors(rectangularB),
    [lattice, rectangularB]
  );

  // Generate lattice points for preview (extended range for more circles)
  const latticePoints = useMemo(() => {
    const points: { x: number; y: number }[] = [];
    const { a1, a2 } = basisVectors;
    
    // Generate a grid of unit cells
    const range = 4;
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
  }, [basisVectors, lattice.atomPositions]);

  // Calculate view bounds
  const viewBounds = useMemo(() => {
    const { a1, a2 } = basisVectors;
    const maxExtent = Math.max(
      Math.abs(a1[0]) + Math.abs(a2[0]),
      Math.abs(a1[1]) + Math.abs(a2[1])
    ) * 2.5;
    return { min: -maxExtent, max: maxExtent, size: maxExtent * 2 };
  }, [basisVectors]);

  const bgColor = getEpsilonColor(backgroundEpsilon);
  const circleColor = getEpsilonColor(circleEpsilon);

  const buttonStyle = (isSelected: boolean) => ({
    padding: '0.6rem 1rem',
    borderRadius: '8px',
    border: isSelected 
      ? '2px solid rgba(100, 200, 255, 0.8)' 
      : '1px solid rgba(255,255,255,0.2)',
    background: isSelected 
      ? 'rgba(100, 200, 255, 0.2)' 
      : 'rgba(255,255,255,0.05)',
    color: 'white',
    cursor: 'pointer',
    fontSize: '0.875rem',
    transition: 'all 0.2s',
    flex: 1,
  });

  return (
    <div style={{
      background: 'rgba(0, 0, 0, 0.4)',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      borderRadius: '24px',
      padding: '2rem',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      maxWidth: '1000px',
      width: '100%',
    }}>
      <h2 style={{
        fontSize: '1.75rem',
        fontWeight: 600,
        color: 'white',
        marginBottom: '1.5rem',
        textAlign: 'center',
      }}>
        Crystal Builder
      </h2>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '2rem',
      }}>
        {/* Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Bravais Lattice Types - 3 in a row */}
          <div>
            <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.875rem', marginBottom: '0.5rem', display: 'block' }}>
              Bravais Lattice
            </label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {(Object.keys(BRAVAIS_LATTICES) as Exclude<LatticeType, 'honeycomb'>[]).map((type) => (
                <button
                  key={type}
                  onClick={() => setLatticeType(type)}
                  style={buttonStyle(latticeType === type)}
                >
                  {BRAVAIS_LATTICES[type].name}
                </button>
              ))}
            </div>
          </div>

          {/* Special Lattices */}
          <div>
            <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.875rem', marginBottom: '0.5rem', display: 'block' }}>
              Special Lattices
            </label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {(Object.keys(SPECIAL_LATTICES) as ('honeycomb')[]).map((type) => (
                <button
                  key={type}
                  onClick={() => setLatticeType(type)}
                  style={buttonStyle(latticeType === type)}
                >
                  {SPECIAL_LATTICES[type].name}
                </button>
              ))}
            </div>
          </div>

          {/* Rectangular parameter slider */}
          {latticeType === 'rectangular' && (
            <div>
              <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.875rem', marginBottom: '0.5rem', display: 'block' }}>
                Parameter b: {rectangularB.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.01"
                value={rectangularB}
                onChange={(e) => setRectangularB(parseFloat(e.target.value))}
                style={{
                  width: '100%',
                  accentColor: 'rgba(100, 200, 255, 0.8)',
                }}
              />
            </div>
          )}

          {/* Circle Radius */}
          <div>
            <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.875rem', marginBottom: '0.5rem', display: 'block' }}>
              Circle Radius: {radius.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.05"
              max="0.5"
              step="0.01"
              value={radius}
              onChange={(e) => setRadius(parseFloat(e.target.value))}
              style={{
                width: '100%',
                accentColor: 'rgba(100, 200, 255, 0.8)',
              }}
            />
          </div>

          {/* Background Epsilon Slider */}
          <div>
            <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.875rem', marginBottom: '0.5rem', display: 'block' }}>
              Background ε: {backgroundEpsilon.toFixed(1)}
            </label>
            <input
              type="range"
              min="1"
              max="13"
              step="0.1"
              value={backgroundEpsilon}
              onChange={(e) => setBackgroundEpsilon(parseFloat(e.target.value))}
              style={{
                width: '100%',
                accentColor: 'rgba(100, 200, 255, 0.8)',
              }}
            />
          </div>

          {/* Circle Epsilon Slider */}
          <div>
            <label style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.875rem', marginBottom: '0.5rem', display: 'block' }}>
              Circle ε: {circleEpsilon.toFixed(1)}
            </label>
            <input
              type="range"
              min="1"
              max="13"
              step="0.1"
              value={circleEpsilon}
              onChange={(e) => setCircleEpsilon(parseFloat(e.target.value))}
              style={{
                width: '100%',
                accentColor: 'rgba(100, 200, 255, 0.8)',
              }}
            />
          </div>

          {/* Epsilon Color Scale */}
          <div style={{
            padding: '1rem',
            background: 'rgba(255,255,255,0.05)',
            borderRadius: '8px',
          }}>
            <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.75rem', marginBottom: '0.5rem' }}>
              Epsilon Color Scale
            </div>
            <div style={{
              height: '20px',
              borderRadius: '4px',
              background: 'linear-gradient(to right, rgb(255, 255, 255) 0%, rgb(40, 40, 40) 100%)',
            }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', marginTop: '0.25rem' }}>
              <span>ε = 1</span>
              <span>ε = 13</span>
            </div>
          </div>
        </div>

        {/* Preview */}
        <div style={{
          background: bgColor,
          borderRadius: '16px',
          overflow: 'hidden',
          aspectRatio: '1',
          position: 'relative',
        }}>
          <svg
            viewBox={`${viewBounds.min} ${viewBounds.min} ${viewBounds.size} ${viewBounds.size}`}
            style={{ width: '100%', height: '100%' }}
          >
            {/* Lattice circles */}
            {latticePoints.map((point, i) => (
              <circle
                key={i}
                cx={point.x}
                cy={-point.y} // Flip y for conventional coords
                r={radius}
                fill={circleColor}
              />
            ))}

            {/* Unit cell outline */}
            <g stroke="rgba(48, 105, 142, 0.5)" strokeWidth="0.03" fill="none" strokeDasharray="0.1">
              <path d={`M 0 0 L ${basisVectors.a1[0]} ${-basisVectors.a1[1]} L ${basisVectors.a1[0] + basisVectors.a2[0]} ${-(basisVectors.a1[1] + basisVectors.a2[1])} L ${basisVectors.a2[0]} ${-basisVectors.a2[1]} Z`} />
            </g>

            {/* Basis vectors with smaller arrowheads */}
            <defs>
              <marker id="arrowhead-a1" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                <polygon points="0 0, 6 2, 0 4" fill="#30698e" />
              </marker>
              <marker id="arrowhead-a2" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                <polygon points="0 0, 6 2, 0 4" fill="#30698e" />
              </marker>
            </defs>
            <g>
              <line
                x1="0" y1="0"
                x2={basisVectors.a1[0]} y2={-basisVectors.a1[1]}
                stroke="#30698e"
                strokeWidth="0.04"
                markerEnd="url(#arrowhead-a1)"
              />
              <text
                x={basisVectors.a1[0] * 0.5}
                y={-basisVectors.a1[1] * 0.5 - 0.15}
                fill="#30698e"
                fontSize="0.25"
                textAnchor="middle"
              >
                a₁
              </text>
            </g>
            <g>
              <line
                x1="0" y1="0"
                x2={basisVectors.a2[0]} y2={-basisVectors.a2[1]}
                stroke="#30698e"
                strokeWidth="0.04"
                markerEnd="url(#arrowhead-a2)"
              />
              <text
                x={basisVectors.a2[0] * 0.5 - 0.15}
                y={-basisVectors.a2[1] * 0.5}
                fill="#30698e"
                fontSize="0.25"
                textAnchor="middle"
              >
                a₂
              </text>
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}
