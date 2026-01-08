'use client';

import { useEffect, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import type { BandData, BandDataMeta } from './types';

type AnimationPhase = 
  | 'idle'
  | 'fadeInLeft'
  | 'drawLeft'
  | 'fadeInRight'
  | 'drawRight'
  | 'fadeOutBackgrounds'
  | 'mergePlots'
  | 'fadeInFinal'
  | 'complete';

// Plot width as percentage of container
const PLOT_WIDTH_PERCENT = 45;
const GAP_PERCENT = 2;

/**
 * Load band data from binary format
 */
async function loadBandData(): Promise<BandData | null> {
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
  try {
    // Fetch metadata
    const metaRes = await fetch(`${basePath}/data/blaze/band_data_meta.json`)
    if (!metaRes.ok) {
      // Fall back to legacy JSON format
      console.warn('Binary format not found, falling back to JSON')
      const jsonRes = await fetch(`${basePath}/band_diagram.json`)
      if (!jsonRes.ok) return null
      return await jsonRes.json()
    }
    const meta: BandDataMeta = await metaRes.json()
    
    // Fetch binary data
    const binRes = await fetch(`${basePath}/data/blaze/band_data.bin`)
    if (!binRes.ok) return null
    const buffer = await binRes.arrayBuffer()
    
    const { n_bands, n_points } = meta
    const floatView = new Float32Array(buffer)
    
    // Parse bands from binary (band-major layout)
    const bands: number[][] = []
    for (let b = 0; b < n_bands; b++) {
      const band: number[] = []
      const offset = b * n_points
      for (let p = 0; p < n_points; p++) {
        band.push(floatView[offset + p])
      }
      bands.push(band)
    }
    
    return {
      bands,
      n_bands,
      n_points,
      symmetry_points: meta.symmetry_points,
      params: meta.params,
    }
  } catch (err) {
    console.error('Failed to load band data:', err)
    return null
  }
}

export default function BandComparisonPlot() {
  const leftCanvasRef = useRef<HTMLCanvasElement>(null);
  const rightCanvasRef = useRef<HTMLCanvasElement>(null);
  const mergedCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [bandData, setBandData] = useState<BandData | null>(null);
  const [phase, setPhase] = useState<AnimationPhase>('idle');
  const [leftProgress, setLeftProgress] = useState(0);
  const [rightProgress, setRightProgress] = useState(0);
  const [backgroundOpacity, setBackgroundOpacity] = useState(1);
  const [mergeProgress, setMergeProgress] = useState(0);
  const [finalOpacity, setFinalOpacity] = useState(0);
  const [restartButtonVisible, setRestartButtonVisible] = useState(false);
  
  const animationRef = useRef<number>(0);

  // Reset animation to beginning
  const restartAnimation = () => {
    cancelAnimationFrame(animationRef.current);
    setPhase('idle');
    setLeftProgress(0);
    setRightProgress(0);
    setBackgroundOpacity(1);
    setMergeProgress(0);
    setFinalOpacity(0);
    setRestartButtonVisible(false);
    // Small delay then start
    setTimeout(() => setPhase('fadeInLeft'), 100);
  };

  // Load band data
  useEffect(() => {
    loadBandData()
      .then(data => {
        if (data) setBandData(data)
        else console.error('Failed to load band data')
      })
      .catch(err => console.error('Failed to load band data:', err));
  }, []);

  // Start animation when visible
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && phase === 'idle') {
          setPhase('fadeInLeft');
        }
      },
      { threshold: 0.3 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [phase]);

  // Phase transitions
  const handleLeftTransitionEnd = () => {
    if (phase === 'fadeInLeft') {
      setPhase('drawLeft');
    }
  };

  const handleRightTransitionEnd = () => {
    if (phase === 'fadeInRight') {
      setPhase('drawRight');
    }
  };

  // Animation loops for each phase
  useEffect(() => {
    if (!bandData) return;

    if (phase === 'drawLeft') {
      const startTime = performance.now();
      const duration = 1000;

      const animate = (timestamp: number) => {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        setLeftProgress(progress);

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setPhase('fadeInRight');
        }
      };

      animationRef.current = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationRef.current);
    }

    if (phase === 'drawRight') {
      const startTime = performance.now();
      const duration = 1000;

      const animate = (timestamp: number) => {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        setRightProgress(progress);

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setTimeout(() => setPhase('fadeOutBackgrounds'), 500);
        }
      };

      animationRef.current = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationRef.current);
    }

    if (phase === 'fadeOutBackgrounds') {
      const startTime = performance.now();
      const duration = 800;

      const animate = (timestamp: number) => {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        setBackgroundOpacity(1 - progress);

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setPhase('mergePlots');
        }
      };

      animationRef.current = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationRef.current);
    }

    if (phase === 'mergePlots') {
      const startTime = performance.now();
      const duration = 1200;

      const animate = (timestamp: number) => {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease in-out cubic for smooth acceleration and deceleration
        const eased = progress < 0.5
          ? 4 * progress * progress * progress
          : 1 - Math.pow(-2 * progress + 2, 3) / 2;
        setMergeProgress(eased);

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setPhase('fadeInFinal');
        }
      };

      animationRef.current = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationRef.current);
    }

    if (phase === 'fadeInFinal') {
      const startTime = performance.now();
      const duration = 600;

      const animate = (timestamp: number) => {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        setFinalOpacity(progress);

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setPhase('complete');
          // Show restart button after a small delay
          setTimeout(() => setRestartButtonVisible(true), 100);
        }
      };

      animationRef.current = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationRef.current);
    }
  }, [phase, bandData]);

  // Draw function for individual canvases
  const drawBands = (
    canvas: HTMLCanvasElement,
    progress: number,
    color: string,
    showAxes: boolean,
    title: string,
    bgOpacity: number
  ) => {
    if (!bandData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 50, right: 30, bottom: 50, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (showAxes && bgOpacity > 0) {
      // Draw plot background
      ctx.fillStyle = `rgba(0, 0, 0, ${0.3 * bgOpacity})`;
      ctx.fillRect(padding.left, padding.top, plotWidth, plotHeight);

      // Draw axes
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.3 * bgOpacity})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, padding.top + plotHeight);
      ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
      ctx.stroke();

      // Title
      ctx.fillStyle = `rgba(255, 255, 255, ${0.9 * bgOpacity})`;
      ctx.font = 'bold 18px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(title, width / 2, 30);

      // Y axis label
      ctx.save();
      ctx.translate(20, padding.top + plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${0.6 * bgOpacity})`;
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('ωa/2πc', 0, 0);
      ctx.restore();

      // X axis label
      ctx.fillStyle = `rgba(255, 255, 255, ${0.6 * bgOpacity})`;
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Wave vector k', padding.left + plotWidth / 2, height - 15);

      // High symmetry points
      const symmetryLabels = bandData.symmetry_points.labels;
      const symmetryIndices = bandData.symmetry_points.indices;
      const nPoints = bandData.n_points;

      symmetryLabels.forEach((point, i) => {
        const x = padding.left + (symmetryIndices[i] / (nPoints - 1)) * plotWidth;
        ctx.fillStyle = `rgba(255, 255, 255, ${0.5 * bgOpacity})`;
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(point, x, padding.top + plotHeight + 20);

        if (i > 0 && i < symmetryLabels.length - 1) {
          ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 * bgOpacity})`;
          ctx.beginPath();
          ctx.moveTo(x, padding.top);
          ctx.lineTo(x, padding.top + plotHeight);
          ctx.stroke();
        }
      });

      // Y axis ticks
      const maxFreq = Math.max(...bandData.bands.flat());
      const yMax = Math.ceil(maxFreq * 10) / 10;
      const nYTicks = 5;

      for (let i = 0; i <= nYTicks; i++) {
        const val = (i / nYTicks) * yMax;
        const y = padding.top + plotHeight - (i / nYTicks) * plotHeight;
        ctx.fillStyle = `rgba(255, 255, 255, ${0.4 * bgOpacity})`;
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(2), padding.left - 8, y + 4);

        if (i > 0) {
          ctx.strokeStyle = `rgba(255, 255, 255, ${0.05 * bgOpacity})`;
          ctx.beginPath();
          ctx.moveTo(padding.left, y);
          ctx.lineTo(padding.left + plotWidth, y);
          ctx.stroke();
        }
      }
    }

    // Draw bands
    const nPoints = bandData.n_points;
    const drawPoints = Math.floor(progress * nPoints);
    const maxFreq = Math.max(...bandData.bands.flat());
    const yMax = Math.ceil(maxFreq * 10) / 10;
    const nBands = bandData.bands.length;

    bandData.bands.forEach((band, bandIdx) => {
      if (drawPoints === 0) return;

      const opacity = 0.95 - (bandIdx / (nBands - 1)) * 0.65;
      ctx.strokeStyle = color.replace('OPACITY', opacity.toString());
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = 0; i < drawPoints && i < band.length; i++) {
        const x = padding.left + (i / (nPoints - 1)) * plotWidth;
        const y = padding.top + plotHeight - (band[i] / yMax) * plotHeight;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });
  };

  // Draw merged canvas (only axes/background, no bands)
  const drawMerged = () => {
    if (!bandData || !mergedCanvasRef.current) return;

    const canvas = mergedCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 50, right: 30, bottom: 50, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    ctx.clearRect(0, 0, width, height);

    // Only draw axes/background when finalOpacity > 0
    if (finalOpacity > 0) {
      // Draw plot background
      ctx.fillStyle = `rgba(0, 0, 0, ${0.3 * finalOpacity})`;
      ctx.fillRect(padding.left, padding.top, plotWidth, plotHeight);

      // Draw axes
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.3 * finalOpacity})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, padding.top + plotHeight);
      ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
      ctx.stroke();

      // Title
      ctx.fillStyle = `rgba(255, 255, 255, ${0.9 * finalOpacity})`;
      ctx.font = 'bold 18px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('MPB & BLAZE 2D', width / 2, 30);

      // Y axis label
      ctx.save();
      ctx.translate(20, padding.top + plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${0.6 * finalOpacity})`;
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('ωa/2πc', 0, 0);
      ctx.restore();

      // X axis label
      ctx.fillStyle = `rgba(255, 255, 255, ${0.6 * finalOpacity})`;
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Wave vector k', padding.left + plotWidth / 2, height - 15);

      // High symmetry points
      const symmetryLabels = bandData.symmetry_points.labels;
      const symmetryIndices = bandData.symmetry_points.indices;
      const nPoints = bandData.n_points;

      symmetryLabels.forEach((point, i) => {
        const x = padding.left + (symmetryIndices[i] / (nPoints - 1)) * plotWidth;
        ctx.fillStyle = `rgba(255, 255, 255, ${0.5 * finalOpacity})`;
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(point, x, padding.top + plotHeight + 20);

        if (i > 0 && i < symmetryLabels.length - 1) {
          ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 * finalOpacity})`;
          ctx.beginPath();
          ctx.moveTo(x, padding.top);
          ctx.lineTo(x, padding.top + plotHeight);
          ctx.stroke();
        }
      });

      // Y axis ticks
      const maxFreq = Math.max(...bandData.bands.flat());
      const yMax = Math.ceil(maxFreq * 10) / 10;
      const nYTicks = 5;

      for (let i = 0; i <= nYTicks; i++) {
        const val = (i / nYTicks) * yMax;
        const y = padding.top + plotHeight - (i / nYTicks) * plotHeight;
        ctx.fillStyle = `rgba(255, 255, 255, ${0.4 * finalOpacity})`;
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(2), padding.left - 8, y + 4);

        if (i > 0) {
          ctx.strokeStyle = `rgba(255, 255, 255, ${0.05 * finalOpacity})`;
          ctx.beginPath();
          ctx.moveTo(padding.left, y);
          ctx.lineTo(padding.left + plotWidth, y);
          ctx.stroke();
        }
      }
    }

    // Don't draw any bands on merged canvas - they persist from side plots
  };

  // Redraw canvases when state changes
  useEffect(() => {
    if (leftCanvasRef.current && bandData) {
      drawBands(
        leftCanvasRef.current,
        leftProgress,
        'rgba(100, 200, 255, OPACITY)',
        true,
        'MPB',
        backgroundOpacity
      );
    }
  }, [bandData, leftProgress, backgroundOpacity]);

  useEffect(() => {
    if (rightCanvasRef.current && bandData) {
      drawBands(
        rightCanvasRef.current,
        rightProgress,
        'rgba(255, 150, 80, OPACITY)',
        true,
        'BLAZE 2D',
        backgroundOpacity
      );
    }
  }, [bandData, rightProgress, backgroundOpacity]);

  const showLeft = phase !== 'idle';
  const showRight = ['fadeInRight', 'drawRight', 'fadeOutBackgrounds', 'mergePlots', 'fadeInFinal', 'complete'].includes(phase);
  const inMergePhase = phase === 'mergePlots' || phase === 'fadeInFinal' || phase === 'complete';

  useEffect(() => {
    if (inMergePhase && bandData) {
      // Only draw axes/background - bands persist from side plots
      drawMerged();
    }
  }, [bandData, mergeProgress, finalOpacity, phase, inMergePhase]);

  // Position calculations using percentages
  // Initial positions: left plot at ~29%, right plot at ~52% (with gap between)
  // Final position: both at 31% (center - half width)
  const leftInitial = (50 - PLOT_WIDTH_PERCENT - GAP_PERCENT / 2);
  const rightInitial = (50 + GAP_PERCENT / 2);
  const centerPos = (50 - PLOT_WIDTH_PERCENT / 2);
  
  // Interpolate positions during merge
  const leftPos = leftInitial + mergeProgress * (centerPos - leftInitial);
  const rightPos = rightInitial + mergeProgress * (centerPos - rightInitial);

  // Side plots: backgrounds fade out, but bands stay visible
  const sideBgOpacity = phase === 'fadeInFinal' || phase === 'complete' ? 1 - finalOpacity : 1;
  // Bands on side plots never fade out
  const sideBandsOpacity = 1;

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        position: 'relative',
        height: '450px',
      }}
    >
      {/* Left plot (MPB - Blue) */}
      <div
        onTransitionEnd={handleLeftTransitionEnd}
        style={{
          position: 'absolute',
          left: `${leftPos}%`,
          top: 0,
          width: `${PLOT_WIDTH_PERCENT}%`,
          height: '100%',
          zIndex: 2,
          background: `rgba(0, 0, 0, ${0.4 * backgroundOpacity * sideBgOpacity})`,
          backdropFilter: backgroundOpacity * sideBgOpacity > 0 ? 'blur(12px)' : 'none',
          WebkitBackdropFilter: backgroundOpacity * sideBgOpacity > 0 ? 'blur(12px)' : 'none',
          borderRadius: '24px',
          padding: '1rem',
          border: `1px solid rgba(255, 255, 255, ${0.1 * backgroundOpacity * sideBgOpacity})`,
          boxSizing: 'border-box',
          opacity: showLeft ? 1 : 0,
          transform: showLeft ? 'translateX(0)' : 'translateX(-40px)',
          transition: phase === 'fadeInLeft' ? 'opacity 0.6s ease-out, transform 0.6s ease-out' : 'none',
          pointerEvents: sideBgOpacity < 0.1 ? 'none' : 'auto',
        }}
      >
        <canvas
          ref={leftCanvasRef}
          style={{
            width: '100%',
            height: '100%',
            borderRadius: '12px',
          }}
        />
      </div>

      {/* Right plot (BLAZE 2D - Orange) */}
      <div
        onTransitionEnd={handleRightTransitionEnd}
        style={{
          position: 'absolute',
          left: `${rightPos}%`,
          top: 0,
          width: `${PLOT_WIDTH_PERCENT}%`,
          height: '100%',
          zIndex: 2,
          background: `rgba(0, 0, 0, ${0.4 * backgroundOpacity * sideBgOpacity})`,
          backdropFilter: backgroundOpacity * sideBgOpacity > 0 ? 'blur(12px)' : 'none',
          WebkitBackdropFilter: backgroundOpacity * sideBgOpacity > 0 ? 'blur(12px)' : 'none',
          borderRadius: '24px',
          padding: '1rem',
          border: `1px solid rgba(255, 255, 255, ${0.1 * backgroundOpacity * sideBgOpacity})`,
          boxSizing: 'border-box',
          opacity: showRight ? 1 : 0,
          transform: showRight ? 'translateX(0)' : 'translateX(40px)',
          transition: phase === 'fadeInRight' ? 'opacity 0.6s ease-out, transform 0.6s ease-out' : 'none',
          pointerEvents: sideBgOpacity < 0.1 ? 'none' : 'auto',
        }}
      >
        <canvas
          ref={rightCanvasRef}
          style={{
            width: '100%',
            height: '100%',
            borderRadius: '12px',
          }}
        />
      </div>

      {/* Merged plot - same size, centered */}
      <div
        style={{
          position: 'absolute',
          left: `${centerPos}%`,
          top: 0,
          width: `${PLOT_WIDTH_PERCENT}%`,
          height: '100%',
          zIndex: 1,
          background: `rgba(0, 0, 0, ${0.4 * finalOpacity})`,
          backdropFilter: finalOpacity > 0 ? 'blur(12px)' : 'none',
          WebkitBackdropFilter: finalOpacity > 0 ? 'blur(12px)' : 'none',
          borderRadius: '24px',
          padding: '1rem',
          border: `1px solid rgba(255, 255, 255, ${0.1 * finalOpacity})`,
          boxSizing: 'border-box',
          opacity: inMergePhase ? 1 : 0,
          pointerEvents: inMergePhase ? 'auto' : 'none',
        }}
      >
        <canvas
          ref={mergedCanvasRef}
          style={{
            width: '100%',
            height: '100%',
            borderRadius: '12px',
          }}
        />
        
        {/* Restart button - attached to final plot */}
        <button
          onClick={restartAnimation}
          style={{
            position: 'absolute',
            top: '0.75rem',
            right: '0.75rem',
            background: 'transparent',
            border: 'none',
            borderRadius: '8px',
            padding: '0.5rem',
            cursor: 'pointer',
            color: 'rgba(255, 255, 255, 0.6)',
            transition: 'all 0.3s ease-out',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: restartButtonVisible ? 1 : 0,
            transform: restartButtonVisible ? 'rotate(0deg)' : 'rotate(360deg)',
            pointerEvents: restartButtonVisible ? 'auto' : 'none',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
            e.currentTarget.style.color = 'rgba(255, 255, 255, 0.9)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = 'rgba(255, 255, 255, 0.6)';
          }}
          title="Restart animation"
        >
          <RotateCcw size={18} />
        </button>
      </div>
    </div>
  );
}
