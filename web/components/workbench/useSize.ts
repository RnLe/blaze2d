'use client';
import { useEffect, useRef, useState } from 'react';

export function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0, dpr: 1 });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      setSize({ width, height, dpr: window.devicePixelRatio || 1 });
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    window.addEventListener('resize', resize);
    resize();
    return () => { observer.disconnect(); window.removeEventListener('resize', resize); };
  }, []);
  return { ref, ...size };
}
