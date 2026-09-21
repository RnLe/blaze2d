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
  saturation: number;    // Per-family, so orange can sit back from blue
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
  
  // The two brand colours rather than one: Blaze blue is hsl(207 58% 46%) and
  // Blaze orange hsl(25 83% 54%). Each family gets a narrow hue spread so the
  // curves still read as one palette, and the orange runs at roughly half its
  // brand saturation -- at full strength it burns through the blue instead of
  // mixing with it.
  const orange = Math.random() < 0.42;
  const hue = orange ? 20 + Math.random() * 14 : 198 + Math.random() * 18;
  const saturation = orange ? 42 + Math.random() * 12 : 54 + Math.random() * 12;

  return {
    bandIndex,
    progress: initialProgress,
    yOffset: 0.1 + Math.random() * 0.8,
    hue,
    saturation,
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

    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    let width = innerWidth, height = innerHeight, previous = 0;
    const resize = () => {
      width = innerWidth; height = innerHeight;
      const ratio = Math.min(devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
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
      const step = previous ? Math.min((timestamp - previous) / (1000 / 60), 2) : 1;
      previous = timestamp;
      // Spawn new bands periodically
      if (timestamp - lastSpawnTime > spawnInterval) {
        spawnBand(0);
        lastSpawnTime = timestamp;
      }

      // Clear canvas completely each frame
      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.fillRect(0, 0, width, height);

      // Sort by depth so far bands render first (behind close bands)
      const sortedBands = [...activeBandsRef.current].sort((a, b) => a.depth - b.depth);

      // Update and draw each active band
      activeBandsRef.current = activeBandsRef.current.filter(band => {
        if (!motion.matches) band.progress += band.speed * step;
        
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
          
          const x = (pointIdx / nPoints) * width;
          const baseY = band.yOffset * height;
          const curveY = bandCurve[pointIdx] * height * 0.3;
          const y = baseY + curveY - height * 0.15;

          // Alpha fades linearly from 1 at head to 0 at trail end
          const alpha = 1 - (distFromHead / band.trailLength);
          
          // Far bands are also slightly dimmer
          const brightnessMultiplier = 0.5 + band.depth * 0.5;

          ctx.beginPath();
          ctx.arc(x, y, band.dotSize, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${band.hue}, ${band.saturation}%, ${44 + band.depth * 14}%, ${alpha * 0.9 * brightnessMultiplier})`;
          ctx.fill();
        }
      }

      if (!motion.matches && !document.hidden) animationRef.current = requestAnimationFrame(animate);
    };

    const resume = () => {
      cancelAnimationFrame(animationRef.current); previous = 0;
      if (!document.hidden) animationRef.current = requestAnimationFrame(animate);
    };
    motion.addEventListener('change', resume);
    document.addEventListener('visibilitychange', resume);
    window.addEventListener('resize', resume);
    resume();

    return () => {
      cancelAnimationFrame(animationRef.current);
      window.removeEventListener('resize', resize);
      window.removeEventListener('resize', resume);
      motion.removeEventListener('change', resume);
      document.removeEventListener('visibilitychange', resume);
    };
  }, [bandData]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
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
