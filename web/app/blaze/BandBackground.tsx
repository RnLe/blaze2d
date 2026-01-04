'use client';

import { useEffect, useRef, useState } from 'react';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

interface BandData {
  bands: number[][];
  n_points: number;
}

interface ActiveBand {
  bandIndex: number;
  progress: number;      // 0 to 1+, how far across the screen
  yOffset: number;       // Random vertical offset (0-1)
  hue: number;           // Color hue
  // Depth-based properties (0 = far/background, 1 = close/foreground)
  depth: number;
  dotSize: number;
  speed: number;
  trailLength: number;
  dotSpacing: number;    // Draw every Nth point
}

// Depth configuration: creates illusion of 3D space
function createBandFromDepth(depth: number, bandIndex: number, initialProgress: number = 0): ActiveBand {
  // depth: 0 = far away (small, slow, long trail), 1 = close (big, fast, short trail)
  
  // Color range: blue (200-240), purple (260-300), neon green (100-140)
  const colorRanges = [
    200 + Math.random() * 40,  // Blue
    260 + Math.random() * 40,  // Purple
    100 + Math.random() * 40,  // Neon green
  ];
  const hue = colorRanges[Math.floor(Math.random() * colorRanges.length)];

  return {
    bandIndex,
    progress: initialProgress,
    yOffset: 0.1 + Math.random() * 0.8,
    hue,
    depth,
    // Far (depth=0): small dots (0.8), slow (0.000192), long trail (120), dense (1)
    // Close (depth=1): big dots (3.2), moderate speed (0.000528), medium trail (50), sparse (2.5)
    dotSize: 0.8 + depth * 2.4,
    speed: 0.000192 + depth * 0.000336,
    trailLength: Math.round(120 - depth * 70),
    dotSpacing: Math.round(1 + depth * 1.5),
  };
}

export default function BandBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [bandData, setBandData] = useState<BandData | null>(null);
  const activeBandsRef = useRef<ActiveBand[]>([]);
  const animationRef = useRef<number>(0);
  const initializedRef = useRef(false);

  // Load band data
  useEffect(() => {
    fetch(`${basePath}/band_curves.json`)
      .then(res => res.json())
      .then(data => setBandData(data))
      .catch(err => console.error('Failed to load band data:', err));
  }, []);

  // Animation loop
  useEffect(() => {
    if (!bandData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Resize canvas to window
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const maxBands = 29;
    const spawnInterval = 1200;
    let lastSpawnTime = 0;

    const spawnBand = (initialProgress: number = 0) => {
      if (activeBandsRef.current.length >= maxBands) return;
      
      const depth = Math.random(); // Random depth for variety
      const bandIndex = Math.floor(Math.random() * bandData.bands.length);
      const newBand = createBandFromDepth(depth, bandIndex, initialProgress);
      activeBandsRef.current.push(newBand);
    };

    // Initialize with bands at random positions (only on first load)
    if (!initializedRef.current) {
      initializedRef.current = true;
      const initialBands = 13;
      for (let i = 0; i < initialBands; i++) {
        const randomProgress = Math.random() * 0.9; // Random position 0-90% across screen
        spawnBand(randomProgress);
      }
    }

    const animate = (timestamp: number) => {
      // Spawn new bands periodically
      if (timestamp - lastSpawnTime > spawnInterval) {
        spawnBand(0);
        lastSpawnTime = timestamp;
      }

      // Clear canvas completely each frame
      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Sort by depth so far bands render first (behind close bands)
      const sortedBands = [...activeBandsRef.current].sort((a, b) => a.depth - b.depth);

      // Update and draw each active band
      activeBandsRef.current = activeBandsRef.current.filter(band => {
        band.progress += band.speed;
        
        // Remove if fully off screen (trail has completely passed)
        const maxTrailProgress = band.trailLength / bandData.n_points;
        if (band.progress > 1.0 + maxTrailProgress + 0.05) return false;

        return true;
      });

      // Draw sorted bands
      for (const band of sortedBands) {
        const bandCurve = bandData.bands[band.bandIndex];
        const nPoints = bandCurve.length;

        // Calculate the head position (leading edge)
        const headIdx = Math.floor(band.progress * nPoints);
        
        // Draw trail behind the head
        // Use fixed point positions (modulo spacing) so dots don't appear to move
        for (let pointIdx = headIdx; pointIdx >= 0 && pointIdx >= headIdx - band.trailLength; pointIdx--) {
          // Only draw at fixed intervals based on absolute position
          if (pointIdx % band.dotSpacing !== 0) continue;
          if (pointIdx >= nPoints) continue;

          const distFromHead = headIdx - pointIdx;
          
          const x = (pointIdx / nPoints) * canvas.width;
          const baseY = band.yOffset * canvas.height;
          const curveY = bandCurve[pointIdx] * canvas.height * 0.3;
          const y = baseY + curveY - canvas.height * 0.15;

          // Alpha fades linearly from 1 at head to 0 at trail end
          const alpha = 1 - (distFromHead / band.trailLength);
          
          // Far bands are also slightly dimmer
          const brightnessMultiplier = 0.5 + band.depth * 0.5;

          ctx.beginPath();
          ctx.arc(x, y, band.dotSize, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${band.hue}, 80%, ${45 + band.depth * 15}%, ${alpha * 0.9 * brightnessMultiplier})`;
          ctx.fill();
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    // Start animation
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [bandData]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        background: 'black',
      }}
    />
  );
}
