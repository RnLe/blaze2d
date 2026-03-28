import BandBackground from './BandBackground';
import BandComparisonPlot from './BandComparisonPlot';
import BandDiagramPlot from './BandDiagramPlot';
import CrystalBuilder from './CrystalBuilder';
import SpeedComparison from './SpeedComparison';
import JobDriverAnimation from './JobDriverAnimation';
import JobDriverStream from './JobDriverStream';
import PyPICard from './PyPICard';
import InteractiveBandDiagram from './InteractiveBandDiagram';

export default function BlazePage() {
  return (
    <>
      <BandBackground />
      <div style={{ minHeight: '100vh', position: 'relative', zIndex: 1 }}>
      
      {/* Hero Section */}
      <section style={{
        position: 'relative',
        zIndex: 1,
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem',
      }}>
        <h1 style={{
          fontSize: 'clamp(6rem, 20vw, 16rem)',
          fontWeight: 200,
          fontFamily: '"Inter", "Helvetica Neue", "Arial", sans-serif',
          letterSpacing: '-0.04em',
          color: 'white',
          textAlign: 'center',
          marginBottom: '0',
          lineHeight: 0.9,
          textShadow: '0 0 80px rgba(100, 200, 255, 0.15)',
        }}>
          BLAZE 2D
        </h1>
        <p style={{
          fontSize: '1.25rem',
          color: 'rgba(255, 255, 255, 0.6)',
          textAlign: 'center',
          maxWidth: '600px',
          marginTop: '2rem',
          fontWeight: 300,
          letterSpacing: '0.02em',
        }}>
          A lightweight 2D Maxwell solver for photonic band structures
        </p>
        <div style={{ marginTop: '3rem', color: 'rgba(255,255,255,0.4)', fontSize: '0.875rem' }}>
          ↓ Scroll to explore ↓
        </div>
      </section>

      {/* Band Comparison Section */}
      <section style={{
        position: 'relative',
        zIndex: 1,
        padding: '4rem 2rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        <h2 style={{
          fontSize: 'clamp(2rem, 5vw, 3.5rem)',
          fontWeight: 600,
          color: 'white',
          textAlign: 'center',
          marginBottom: '1rem',
          letterSpacing: '-0.02em',
        }}>
          Built on the Shoulders of Giants
        </h2>
        <p style={{
          fontSize: '1.1rem',
          color: 'rgba(255, 255, 255, 0.5)',
          textAlign: 'center',
          maxWidth: '100%',
          marginBottom: '3rem',
          whiteSpace: 'nowrap',
        }}>
          Validated against MIT Photonic Bands — the gold standard in photonic band structure computation
        </p>
        <BandComparisonPlot />
      </section>

      {/* Speed Comparison Section */}
      <section style={{
        position: 'relative',
        zIndex: 1,
        padding: '4rem 2rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        <h2 style={{
          fontSize: 'clamp(2rem, 5vw, 3.5rem)',
          fontWeight: 600,
          color: 'white',
          textAlign: 'center',
          marginBottom: '1rem',
          letterSpacing: '-0.02em',
        }}>
          BLAZING Fast
        </h2>
        <SpeedComparison />
      </section>

      {/* Interactive Band Structure Section */}
      <section style={{
        position: 'relative',
        zIndex: 1,
        padding: '4rem 2rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        <h2 style={{
          fontSize: 'clamp(2rem, 5vw, 3.5rem)',
          fontWeight: 600,
          color: 'white',
          textAlign: 'center',
          marginBottom: '1rem',
          letterSpacing: '-0.02em',
        }}>
          Try It Yourself
        </h2>
        <p style={{
          fontSize: '1.1rem',
          color: 'rgba(255, 255, 255, 0.5)',
          textAlign: 'center',
          marginBottom: '3rem',
          whiteSpace: 'nowrap',
        }}>
          Compute photonic band structures directly in your browser — no installation required
        </p>
        <InteractiveBandDiagram />
      </section>

      {/* Job Driver Animation Section */}
      <section style={{
        position: 'relative',
        zIndex: 1,
        padding: '4rem 2rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        <h2 style={{
          fontSize: 'clamp(2rem, 5vw, 3.5rem)',
          fontWeight: 600,
          color: 'white',
          textAlign: 'center',
          marginBottom: '1rem',
          letterSpacing: '-0.02em',
        }}>
          Smart Job Driver
        </h2>
        <p style={{
          fontSize: '1.1rem',
          color: 'rgba(255, 255, 255, 0.5)',
          textAlign: 'center',
          maxWidth: '100%',
          marginBottom: '3rem',
          whiteSpace: 'nowrap',
        }}>
          Centralized handling of solving and I/O jobs ensures maximum efficiency and performance
        </p>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '3rem',
          justifyContent: 'center',
          alignItems: 'center',
        }}>
          <JobDriverAnimation />
          <div style={{
            fontSize: '3rem',
            fontWeight: 700,
            background: 'linear-gradient(135deg, #64c8ff, #00ff88)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            padding: '0 1rem',
          }}>
            OR
          </div>
          <JobDriverStream />
        </div>
      </section>

      {/* PyPI Section */}
      <section style={{
        position: 'relative',
        zIndex: 1,
        padding: '4rem 2rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        <PyPICard />
      </section>

      </div>
    </>
  );
}

