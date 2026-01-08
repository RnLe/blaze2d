'use client';

import { useState, useEffect, useRef } from 'react';

export default function SpeedComparison() {
  const [mpbCount, setMpbCount] = useState(0);
  const [fastCount, setFastCount] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  // Start counters when section becomes visible
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !hasStarted) {
          setHasStarted(true);
        }
      },
      { threshold: 0.3 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, [hasStarted]);

  // Run counters once started
  useEffect(() => {
    if (!hasStarted) return;

    // MPB counter: increments once every 2 seconds
    const mpbInterval = setInterval(() => {
      setMpbCount(prev => prev + 1);
    }, 2000);

    // Fast counter: increments 5x faster (once every 400ms)
    const fastInterval = setInterval(() => {
      setFastCount(prev => prev + 1);
    }, 2000 / 5);

    return () => {
      clearInterval(mpbInterval);
      clearInterval(fastInterval);
    };
  }, [hasStarted]);

  const cardStyle = {
    textAlign: 'center' as const,
    padding: '1.5rem 2.5rem',
    background: 'rgba(0, 0, 0, 0.3)',
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  };

  const valueStyle = {
    fontSize: '2.5rem',
    fontWeight: 800,
    background: 'linear-gradient(135deg, #64c8ff, #a064ff, #64ffaa)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  };

  const labelStyle = {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: '0.875rem',
    marginTop: '0.5rem',
  };

  return (
    <section
      ref={sectionRef}
      style={{
        position: 'relative',
        zIndex: 1,
        padding: '6rem 2rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '3rem',
      }}
    >
      {/* Speed comparison header */}
      <div style={{ textAlign: 'center' }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: 700,
          color: 'white',
          marginBottom: '0.5rem',
        }}>
          Up to 5× faster than MPB*
        </h2>
        <p style={{
          fontSize: '0.75rem',
          color: 'rgba(255, 255, 255, 0.4)',
          marginTop: '0.5rem',
        }}>
          *for selected polarizations and high-throughput workloads
        </p>
      </div>

      {/* All stats cards in one row */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '2rem',
        justifyContent: 'center',
        alignItems: 'center',
      }}>
        {/* Time comparison cards */}
        <div style={cardStyle}>
          <div style={valueStyle}>1 hr</div>
          <div style={labelStyle}>MPB</div>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          color: 'rgba(255, 255, 255, 0.3)',
          fontSize: '2rem',
        }}>
          →
        </div>

        <div style={cardStyle}>
          <div style={valueStyle}>12 min</div>
          <div style={labelStyle}>BLAZE 2D</div>
        </div>

        {/* Spacer between time and counters */}
        <div style={{ width: '2rem' }} />

        {/* Live counter cards */}
        <div style={cardStyle}>
          <div style={{
            ...valueStyle,
            fontVariantNumeric: 'tabular-nums',
            minWidth: '120px',
          }}>
            {mpbCount.toLocaleString()}
          </div>
          <div style={labelStyle}>MPB Band Diagrams</div>
        </div>

        <div style={cardStyle}>
          <div style={{
            ...valueStyle,
            fontVariantNumeric: 'tabular-nums',
            minWidth: '120px',
          }}>
            {fastCount.toLocaleString()}
          </div>
          <div style={labelStyle}>BLAZE 2D Band Diagrams</div>
        </div>
      </div>
    </section>
  );
}
