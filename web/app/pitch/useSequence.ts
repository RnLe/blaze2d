'use client';
import { useEffect, useRef, useState } from 'react';
export function useSequence(duration: number, ready = true, resetKey = '') {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false), [elapsed, setElapsed] = useState(0), [revision, setRevision] = useState(0);
  useEffect(() => {
    const observer = new IntersectionObserver(entries => { if (entries.some(entry => entry.isIntersecting)) { setVisible(true); observer.disconnect(); } }, { threshold: .2 });
    if (ref.current) observer.observe(ref.current); return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible || !ready) return;
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const start = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const next = motion.matches ? duration : Math.min(duration, now - start);
      setElapsed(next); if (next < duration) frame = requestAnimationFrame(tick);
    };
    setElapsed(0); frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [visible, ready, duration, revision, resetKey]);
  return { ref, elapsed, replay: () => setRevision(value => value + 1) };
}
export function phase(elapsed: number, start: number, duration: number) { return Math.min(1, Math.max(0, (elapsed - start) / duration)); }
export function smooth(value: number) { return value * value * (3 - 2 * value); }
