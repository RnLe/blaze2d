'use client';

import { useState, useEffect, useRef } from 'react';

export default function JobDriverAnimation() {
  const [calculatedBands, setCalculatedBands] = useState(0);
  const [storedBandsTop, setStoredBandsTop] = useState(0);
  const [storedBandsBottom, setStoredBandsBottom] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);
  // Animation phases: 'idle' | 'fadeIn' | 'moving' | 'fadeOut'
  const [animPhase, setAnimPhase] = useState<'idle' | 'fadeIn' | 'moving' | 'fadeOut'>('idle');
  
  // Use refs for the actual counter values to avoid StrictMode issues
  const calculatedBandsRef = useRef(0);
  const storedBandsTopRef = useRef(0);
  const storedBandsBottomRef = useRef(0);
  // Random threshold for transfer (400-600)
  const transferThresholdRef = useRef(Math.floor(Math.random() * 201) + 400);
  const sectionRef = useRef<HTMLDivElement>(null);
  // Ref to track if transfer is in progress
  const isTransferringRef = useRef(false);

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

  // Run counters once started
  useEffect(() => {
    if (!hasStarted) return;

    // Calculate bands: 112 per second
    const calcInterval = setInterval(() => {
      // Increment calculated bands
      calculatedBandsRef.current += 1;
      setCalculatedBands(calculatedBandsRef.current);
      
      // Increment stored bands (top)
      storedBandsTopRef.current += 1;
      setStoredBandsTop(storedBandsTopRef.current);
      
      // Check if we should trigger transfer
      if (storedBandsTopRef.current >= transferThresholdRef.current && !isTransferringRef.current) {
        isTransferringRef.current = true;
        const transferAmount = storedBandsTopRef.current;
        
        // Reset top counter immediately
        storedBandsTopRef.current = 0;
        setStoredBandsTop(0);
        
        // Phase 1: Fade in
        setAnimPhase('fadeIn');
        
        setTimeout(() => {
          // Phase 2: Move down
          setAnimPhase('moving');
          
          setTimeout(() => {
            // Phase 3: Fade out + transfer the amount
            storedBandsBottomRef.current += transferAmount;
            setStoredBandsBottom(storedBandsBottomRef.current);
            setAnimPhase('fadeOut');
            
            // Set new random threshold for next transfer
            transferThresholdRef.current = Math.floor(Math.random() * 201) + 400;
            
            setTimeout(() => {
              // Phase 4: Reset to idle
              setAnimPhase('idle');
              isTransferringRef.current = false;
            }, 300);
          }, 800);
        }, 150);
      }
    }, 1000 / 112);

    return () => {
      clearInterval(calcInterval);
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

  // File icon SVG
  const FileIcon = () => (
    <svg
      width="32"
      height="32"
      viewBox="0 0 24 24"
      fill="none"
      stroke="url(#fileGradient)"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <defs>
        <linearGradient id="fileGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#64c8ff" />
          <stop offset="100%" stopColor="#00ff88" />
        </linearGradient>
      </defs>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );

  // Hard disk icon SVG
  const HardDiskIcon = () => (
    <svg
      width="48"
      height="48"
      viewBox="0 0 24 24"
      fill="none"
      stroke="url(#diskGradient)"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <defs>
        <linearGradient id="diskGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#64c8ff" />
          <stop offset="100%" stopColor="#00ff88" />
        </linearGradient>
      </defs>
      <rect x="2" y="6" width="20" height="12" rx="2" />
      <circle cx="6" cy="12" r="2" />
      <line x1="12" y1="12" x2="18" y2="12" />
      <line x1="12" y1="10" x2="18" y2="10" />
      <line x1="12" y1="14" x2="18" y2="14" />
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
      {/* Top Card - Batch */}
      <div style={cardStyle}>
        <div style={titleStyle}>Batch</div>
        
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
            {storedBandsTop.toLocaleString()}
          </div>
          <div style={labelStyle}>Stored bands</div>
        </div>
      </div>

      {/* Connection line with animated file */}
      <div style={{
        position: 'relative',
        height: '150px',
        width: '4px',
        background: 'linear-gradient(180deg, rgba(100, 200, 255, 0.3), rgba(0, 255, 136, 0.3))',
        borderRadius: '2px',
        marginTop: '-16px',
        marginBottom: '-16px',
      }}>
        {/* Animated file icon */}
        <div
          style={{
            position: 'absolute',
            left: '50%',
            transform: 'translateX(-50%)',
            top: (animPhase === 'moving' || animPhase === 'fadeOut') ? 'calc(100% - 16px)' : '-16px',
            opacity: (animPhase === 'fadeIn' || animPhase === 'moving') ? 1 : 0,
            transition: animPhase === 'moving' 
              ? 'top 0.8s cubic-bezier(0.4, 0, 0.2, 1)' 
              : animPhase === 'fadeIn'
                ? 'opacity 0.15s ease-in'
                : animPhase === 'fadeOut'
                  ? 'opacity 0.3s ease-out'
                  : 'none',
          }}
        >
          <FileIcon />
        </div>
        
        {/* Glow effect on the line when transferring */}
        <div
          style={{
            position: 'absolute',
            inset: '-2px',
            background: 'linear-gradient(180deg, #64c8ff, #00ff88)',
            borderRadius: '4px',
            opacity: (animPhase === 'moving' || animPhase === 'fadeIn') ? 0.6 : 0,
            transition: 'opacity 0.3s ease-in-out',
            filter: 'blur(4px)',
          }}
        />
      </div>

      {/* Bottom Card - Hard Disk */}
      <div style={{
        ...cardStyle,
        display: 'flex',
        alignItems: 'center',
        gap: '1.5rem',
      }}>
        <HardDiskIcon />
        <div>
          <div style={{
            ...titleStyle,
            marginBottom: '0.5rem',
          }}>
            Hard Disk
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem' }}>
            <div style={valueStyle}>{storedBandsBottom.toLocaleString()}</div>
            <div style={labelStyle}>Stored bands</div>
          </div>
        </div>
      </div>
    </div>
  );
}
