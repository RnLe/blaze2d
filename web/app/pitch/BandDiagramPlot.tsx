'use client';

import { useEffect, useRef, useState } from 'react';
import type { BandData, BandDataMeta } from './types';

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

export default function BandDiagramPlot() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [bandData, setBandData] = useState<BandData | null>(null);
  const [progress, setProgress] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const [fadeInComplete, setFadeInComplete] = useState(false);
  const animationRef = useRef<number>(0);

  // Load band data
  useEffect(() => {
    loadBandData()
      .then(data => {
        if (data) setBandData(data)
        else console.error('Failed to load band data')
      })
      .catch(err => console.error('Failed to load band data:', err));
  }, []);

  // Start fade-in when section becomes visible
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !isVisible) {
          setIsVisible(true);
        }
      },
      { threshold: 0.3 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [isVisible]);

  // Handle fade-in completion to trigger band animation
  const handleTransitionEnd = () => {
    if (isVisible) {
      setFadeInComplete(true);
    }
  };

  // Animation loop for bands (starts after fade-in completes)
  useEffect(() => {
    if (!fadeInComplete || !bandData || !canvasRef.current) return;

    const startTime = performance.now();
    const duration = 1000; // 1 second to draw

    const animate = (timestamp: number) => {
      const elapsed = timestamp - startTime;
      const newProgress = Math.min(elapsed / duration, 1);
      setProgress(newProgress);

      if (newProgress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationRef.current);
  }, [fadeInComplete, bandData]);

  // Draw the plot
  useEffect(() => {
    if (!bandData || !canvasRef.current) return;

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
    const padding = { top: 30, right: 30, bottom: 50, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.clearRect(0, 0, width, height);

    // Draw plot background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(padding.left, padding.top, plotWidth, plotHeight);

    // Draw axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    // Y axis
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    // X axis
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    // Y axis label
    ctx.save();
    ctx.translate(20, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('ωa/2πc', 0, 0);
    ctx.restore();

    // X axis label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Wave vector k', padding.left + plotWidth / 2, height - 15);

    // High symmetry points from data
    const symmetryLabels = bandData.symmetry_points.labels;
    const symmetryIndices = bandData.symmetry_points.indices;
    const nPoints = bandData.n_points;
    
    symmetryLabels.forEach((point, i) => {
      const x = padding.left + (symmetryIndices[i] / (nPoints - 1)) * plotWidth;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(point, x, padding.top + plotHeight + 20);
      
      // Vertical grid lines at symmetry points
      if (i > 0 && i < symmetryLabels.length - 1) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotHeight);
        ctx.stroke();
      }
    });

    // Find max frequency for Y axis scaling
    const maxFreq = Math.max(...bandData.bands.flat());
    const yMax = Math.ceil(maxFreq * 10) / 10; // Round up to nearest 0.1

    // Y axis ticks
    const nYTicks = 5;
    for (let i = 0; i <= nYTicks; i++) {
      const val = (i / nYTicks) * yMax;
      const y = padding.top + plotHeight - (i / nYTicks) * plotHeight;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(2), padding.left - 8, y + 4);
      
      // Horizontal grid lines
      if (i > 0) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + plotWidth, y);
        ctx.stroke();
      }
    }

    // Draw bands with animation progress
    const drawPoints = Math.floor(progress * nPoints);

    // Blue color with gradient transparency (lowest band = most opaque)
    const nBands = bandData.bands.length;

    bandData.bands.forEach((band, bandIdx) => {
      if (drawPoints === 0) return;

      // Opacity decreases for higher bands: 0.95 for first band, down to 0.3 for last
      const opacity = 0.95 - (bandIdx / (nBands - 1)) * 0.65;
      ctx.strokeStyle = `rgba(100, 200, 255, ${opacity})`;
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = 0; i < drawPoints && i < band.length; i++) {
        const x = padding.left + (i / (nPoints - 1)) * plotWidth;
        // Scale frequency to plot height using actual max frequency
        const y = padding.top + plotHeight - (band[i] / yMax) * plotHeight;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });

  }, [bandData, progress]);

  return (
    <div
      ref={containerRef}
      onTransitionEnd={handleTransitionEnd}
      style={{
        background: 'rgba(0, 0, 0, 0.4)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        borderRadius: '24px',
        padding: '2rem',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        maxWidth: '800px',
        width: '100%',
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? 'translateX(0)' : 'translateX(-40px)',
        transition: 'opacity 0.6s ease-out, transform 0.6s ease-out',
      }}
    >
      <h2 style={{
        fontSize: '1.75rem',
        fontWeight: 600,
        color: 'white',
        marginBottom: '1.5rem',
        textAlign: 'center',
      }}>
        Band Structure
      </h2>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '400px',
          borderRadius: '12px',
        }}
      />
    </div>
  );
}
