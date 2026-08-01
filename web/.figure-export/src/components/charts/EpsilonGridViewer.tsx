'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';

interface EpsilonGridData {
  resolution: number;
  nx: number;
  ny: number;
  epsilon_background: number;
  epsilon_hole: number;
  radius: number;
  grid: number[];  // Flat array, row-major
}

interface EpsilonGridViewerProps {
  initialResolution?: number;
  size?: number;
  showSlider?: boolean;
}

// Custom colors
const COLOR_EPS_HIGH = { r: 0x43, g: 0x5f, b: 0x9d };  // #435f9d for ε=13
const COLOR_EPS_LOW = { r: 0xbb, g: 0xc1, b: 0xcb };   // #bbc1cb for ε=1

// Powers of 2 to mark on slider
const POWER_OF_2_MARKS = [16, 32, 64, 128, 256];

// Interpolate between two colors
function interpolateColor(t: number): [number, number, number] {
  // t=0 → low epsilon (hole), t=1 → high epsilon (background)
  const r = Math.round(COLOR_EPS_LOW.r + t * (COLOR_EPS_HIGH.r - COLOR_EPS_LOW.r));
  const g = Math.round(COLOR_EPS_LOW.g + t * (COLOR_EPS_HIGH.g - COLOR_EPS_LOW.g));
  const b = Math.round(COLOR_EPS_LOW.b + t * (COLOR_EPS_HIGH.b - COLOR_EPS_LOW.b));
  return [r, g, b];
}

// Cache for loaded data and pending requests
const dataCache = new Map<number, EpsilonGridData>();
const promiseCache = new Map<number, Promise<EpsilonGridData>>();

// All supported resolutions (16 to 256, step 16)
const ALL_RESOLUTIONS = Array.from({ length: 16 }, (_, i) => (i + 1) * 16);

import { getAssetPath } from '../../lib/paths';

// Helper to load data with caching and deduplication
function loadResolution(res: number): Promise<EpsilonGridData> {
  if (dataCache.has(res)) return Promise.resolve(dataCache.get(res)!);
  if (promiseCache.has(res)) return promiseCache.get(res)!;

  const promise = fetch(getAssetPath(`/data/epsilon-grids/epsilon-res${res}.json`))
    .then(r => {
      if (!r.ok) throw new Error(`Failed to load epsilon grid (res=${res})`);
      return r.json();
    })
    .then((data: EpsilonGridData) => {
      dataCache.set(res, data);
      promiseCache.delete(res);
      return data;
    });
    
  promiseCache.set(res, promise);
  return promise;
}

export default function EpsilonGridViewer({
  initialResolution = 64,
  size = 300,
  showSlider = true,
}: EpsilonGridViewerProps) {
  // Snap to nearest valid resolution (multiple of 16, 16-256)
  const snapResolution = useCallback((r: number) => {
    const snapped = Math.round(r / 16) * 16;
    return Math.min(256, Math.max(16, snapped));
  }, []);

  const [resolution, setResolution] = useState(() => snapResolution(initialResolution));
  const [data, setData] = useState<EpsilonGridData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Prefetch all data on mount
  useEffect(() => {
    // Start loading all resolutions in the background
    ALL_RESOLUTIONS.forEach(res => {
      loadResolution(res).catch(err => {
        console.warn(`Failed to prefetch resolution ${res}:`, err);
      });
    });
  }, []);

  // Load data for current resolution
  useEffect(() => {
    const validRes = snapResolution(resolution);
    let mounted = true;

    // Build immediate response if available
    if (dataCache.has(validRes)) {
      setData(dataCache.get(validRes)!);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    
    loadResolution(validRes)
      .then(jsonData => {
        if (mounted) {
          setData(jsonData);
          setLoading(false);
        }
      })
      .catch(err => {
        if (mounted) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, [resolution, snapResolution]);

  // Generate image data URL from grid
  const imageDataUrl = useMemo(() => {
    if (!data) return null;
    
    const { nx, ny, grid, epsilon_background, epsilon_hole } = data;
    
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    
    const imageData = ctx.createImageData(nx, ny);
    
    // Normalize values
    const minEps = epsilon_hole;
    const maxEps = epsilon_background;
    const range = maxEps - minEps;
    
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const eps = grid[iy * nx + ix];
        const t = range > 0 ? (eps - minEps) / range : 0;
        const [r, g, b] = interpolateColor(t);
        
        // Flip vertically (canvas y=0 is top, grid iy=0 is bottom)
        const canvasY = ny - 1 - iy;
        const idx = (canvasY * nx + ix) * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  }, [data]);

  // Handle slider change
  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    setResolution(snapResolution(value));
  }, [snapResolution]);

  // Thumb radius for alignment compensation
  const thumbRadius = 7;

  // Calculate tick positions for powers of 2
  // Account for thumb radius: the slider track is effectively shorter by thumbRadius on each side
  const tickPositions = useMemo(() => {
    return POWER_OF_2_MARKS.map(val => ({
      value: val,
      percent: ((val - 16) / (256 - 16)) * 100
    }));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
      {/* Grid display */}
      <div 
        style={{ 
          width: size, 
          height: size, 
          position: 'relative',
          backgroundColor: '#f0f0f0',
          borderRadius: '4px',
          overflow: 'hidden',
        }}
      >
        {loading && (
          <div style={{ 
            position: 'absolute', 
            inset: 0, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            color: '#666',
            fontSize: '14px',
          }}>
            Loading...
          </div>
        )}
        {error && (
          <div style={{ 
            position: 'absolute', 
            inset: 0, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            color: '#800',
            fontSize: '12px',
            padding: '16px',
            textAlign: 'center',
          }}>
            {error}
          </div>
        )}
        {imageDataUrl && (
          <img
            src={imageDataUrl}
            alt={`Smoothed epsilon grid (${data?.resolution}×${data?.resolution})`}
            style={{
              width: '100%',
              height: '100%',
              imageRendering: 'pixelated',
              opacity: loading ? 0.5 : 1,
              transition: 'opacity 0.15s ease',
            }}
          />
        )}
      </div>

      {/* Slider */}
      {showSlider && (
        <div style={{ width: size, position: 'relative' }}>
          {/* Slider input */}
          <input
            type="range"
            min={16}
            max={256}
            step={16}
            value={resolution}
            onChange={handleSliderChange}
            style={{
              width: '100%',
              height: '4px',
              appearance: 'none',
              backgroundColor: '#e0e0e0',
              borderRadius: '2px',
              outline: 'none',
              cursor: 'pointer',
            }}
          />

          {/* Labels for powers of 2 - positioned to align with thumb center */}
          <div style={{ 
            position: 'relative', 
            height: '20px', 
            marginTop: '4px',
            marginLeft: thumbRadius,
            marginRight: thumbRadius,
          }}>
            {tickPositions.map(({ value, percent }) => (
              <span
                key={value}
                style={{
                  position: 'absolute',
                  left: `${percent}%`,
                  transform: 'translateX(-50%)',
                  fontSize: '10px',
                  color: resolution === value ? '#435f9d' : '#999',
                  fontWeight: resolution === value ? 600 : 400,
                  transition: 'color 0.15s ease',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {value}
              </span>
            ))}
          </div>

          {/* Current resolution display */}
          <div style={{
            textAlign: 'center',
            marginTop: '8px',
            fontSize: '12px',
            color: '#666',
          }}>
            Resolution: <span style={{ fontWeight: 600, color: '#a3befa' }}>{resolution}×{resolution}</span>
          </div>
        </div>
      )}

      {/* Custom slider styles */}
      <style>{`
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 14px;
          height: 14px;
          background: #435f9d;
          border-radius: 50%;
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.2);
          transition: transform 0.1s ease;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
          transform: scale(1.1);
        }
        input[type="range"]::-moz-range-thumb {
          width: 14px;
          height: 14px;
          background: #435f9d;
          border-radius: 50%;
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
      `}</style>
    </div>
  );
}
