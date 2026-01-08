'use client';

import { useState, useEffect, useRef } from 'react';

export default function JobDriverStream() {
  const [calculatedBands, setCalculatedBands] = useState(0);
  const [backlog, setBacklog] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);
  const [isClientActive, setIsClientActive] = useState(true);
  
  // Use refs for the actual counter values to avoid StrictMode issues
  const calculatedBandsRef = useRef(0);
  const backlogRef = useRef(0);
  const sectionRef = useRef<HTMLDivElement>(null);
  // Ref to track client active state for interval callbacks
  const isClientActiveRef = useRef(true);
  // Ref for draining interval
  const drainIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Start animation when section becomes visible
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

  // Run counters once started (70% speed = 78.4 per second)
  useEffect(() => {
    if (!hasStarted) return;

    const calcInterval = setInterval(() => {
      // Increment calculated bands
      calculatedBandsRef.current += 1;
      setCalculatedBands(calculatedBandsRef.current);
      
      // If client is inactive, add to backlog
      if (!isClientActiveRef.current) {
        backlogRef.current += 1;
        setBacklog(backlogRef.current);
      }
    }, 1000 / 78.4); // 70% of 112

    return () => {
      clearInterval(calcInterval);
    };
  }, [hasStarted]);

  // Handle client active/inactive toggling
  useEffect(() => {
    if (!hasStarted) return;

    const scheduleNextToggle = () => {
      // Random interval between 3-8 seconds for active, 2.5-7.5 seconds for inactive
      const isActive = isClientActiveRef.current;
      const delay = isActive 
        ? Math.random() * 5000 + 3000  // 3-8 seconds active
        : Math.random() * 5000 + 2500; // 2.5-7.5 seconds inactive
      
      return setTimeout(() => {
        const wasActive = isClientActiveRef.current;
        isClientActiveRef.current = !wasActive;
        setIsClientActive(!wasActive);
        
        // If becoming active and there's backlog, start draining
        if (!wasActive && backlogRef.current > 0) {
          // Clear any existing drain interval
          if (drainIntervalRef.current) {
            clearInterval(drainIntervalRef.current);
          }
          
          // Drain at 5x speed (392 per second) - half as fast as before
          drainIntervalRef.current = setInterval(() => {
            if (backlogRef.current > 0) {
              backlogRef.current -= 1;
              setBacklog(backlogRef.current);
            } else {
              if (drainIntervalRef.current) {
                clearInterval(drainIntervalRef.current);
                drainIntervalRef.current = null;
              }
            }
          }, 1000 / 392);
        }
        
        // Schedule next toggle
        timeoutRef.current = scheduleNextToggle();
      }, delay);
    };

    const timeoutRef = { current: scheduleNextToggle() };

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (drainIntervalRef.current) {
        clearInterval(drainIntervalRef.current);
      }
    };
  }, [hasStarted]);

  const cardStyle = {
    textAlign: 'center' as const,
    padding: '2rem 3rem',
    background: 'rgba(0, 0, 0, 0.3)',
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
    borderRadius: '16px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    minWidth: '320px',
  };

  const titleStyle = {
    fontSize: '1.5rem',
    fontWeight: 700,
    background: 'linear-gradient(135deg, #64c8ff, #00ff88)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    marginBottom: '1.5rem',
  };

  const valueStyle = {
    fontSize: '2.5rem',
    fontWeight: 800,
    background: 'linear-gradient(135deg, #64c8ff, #00ff88)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    fontVariantNumeric: 'tabular-nums' as const,
    minWidth: '180px',
    display: 'inline-block',
  };

  const labelStyle = {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: '0.875rem',
    marginTop: '0.25rem',
  };

  const counterRowStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1rem',
  };

  // Status dot component
  const StatusDot = ({ active }: { active: boolean }) => (
    <div
      style={{
        width: '12px',
        height: '12px',
        borderRadius: '50%',
        background: active ? '#00ff88' : '#ff4444',
        boxShadow: active 
          ? '0 0 8px #00ff88, 0 0 16px #00ff88' 
          : '0 0 8px #ff4444',
        animation: active ? 'pulse-green 2s ease-in-out infinite' : 'none',
      }}
    />
  );

  // Client icon SVG
  const ClientIcon = () => (
    <svg
      width="48"
      height="48"
      viewBox="0 0 24 24"
      fill="none"
      stroke="url(#clientGradient)"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <defs>
        <linearGradient id="clientGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#64c8ff" />
          <stop offset="100%" stopColor="#00ff88" />
        </linearGradient>
      </defs>
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <line x1="8" y1="21" x2="16" y2="21" />
      <line x1="12" y1="17" x2="12" y2="21" />
    </svg>
  );

  return (
    <div
      ref={sectionRef}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '2rem',
        position: 'relative',
        padding: '2rem',
      }}
    >
      {/* CSS for pulsing animation and particle flow */}
      <style>{`
        @keyframes pulse-green {
          0%, 100% { 
            box-shadow: 0 0 8px #00ff88, 0 0 16px #00ff88;
            opacity: 1;
          }
          50% { 
            box-shadow: 0 0 4px #00ff88, 0 0 8px #00ff88;
            opacity: 0.7;
          }
        }
        @keyframes particle-flow {
          0% {
            top: -8px;
            opacity: 0;
          }
          10% {
            opacity: 1;
          }
          90% {
            opacity: 1;
          }
          100% {
            top: calc(100% - 8px);
            opacity: 0;
          }
        }
      `}</style>

      {/* Top Card - Stream */}
      <div style={cardStyle}>
        <div style={titleStyle}>Stream</div>
        
        <div style={counterRowStyle}>
          <div style={valueStyle}>{calculatedBands.toLocaleString()}</div>
          <div style={labelStyle}>Calculated bands</div>
        </div>
        
        <div style={{
          ...counterRowStyle,
          marginBottom: 0,
          paddingTop: '1rem',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        }}>
          <div style={{
            ...valueStyle,
            fontSize: '2rem',
          }}>
            {backlog.toLocaleString()}
          </div>
          <div style={labelStyle}>Backlog</div>
        </div>
      </div>

      {/* Connection - flowing particles */}
      <div style={{
        position: 'relative',
        height: '150px',
        width: '20px',
        marginTop: '-16px',
        marginBottom: '-16px',
        display: 'flex',
        justifyContent: 'center',
      }}>
        {/* Base line (subtle, always visible when active) */}
        <div
          style={{
            position: 'absolute',
            width: '2px',
            height: '100%',
            background: 'linear-gradient(180deg, rgba(100, 200, 255, 0.2), rgba(0, 255, 136, 0.2))',
            borderRadius: '2px',
            opacity: isClientActive ? 1 : 0,
            transition: 'opacity 0.3s ease-in-out',
          }}
        />
        {/* Flowing particles */}
        {[0, 1, 2, 3, 4].map((i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: `linear-gradient(135deg, #64c8ff, #00ff88)`,
              boxShadow: '0 0 8px #64c8ff, 0 0 12px #00ff88',
              animationName: isClientActive ? 'particle-flow' : 'none',
              animationDuration: '3s',
              animationTimingFunction: 'ease-in-out',
              animationIterationCount: 'infinite',
              animationDelay: `${i * 0.6}s`,
              opacity: isClientActive ? 1 : 0,
              transition: 'opacity 0.3s ease-in-out',
            }}
          />
        ))}
      </div>

      {/* Bottom Card - Client */}
      <div style={{
        ...cardStyle,
        display: 'flex',
        alignItems: 'center',
        gap: '1.5rem',
      }}>
        <ClientIcon />
        <div>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '0.5rem',
          }}>
            <div style={{
              ...titleStyle,
              marginBottom: 0,
            }}>
              Client
            </div>
            <StatusDot active={isClientActive} />
          </div>
          <div style={{ 
            color: 'rgba(255, 255, 255, 0.4)', 
            fontSize: '0.875rem',
          }}>
            Python, WebUI, ...
          </div>
        </div>
      </div>
    </div>
  );
}
