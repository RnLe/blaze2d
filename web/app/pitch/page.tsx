import BandBackground from './BandBackground';
import BandComparisonPlot from './BandComparisonPlot';
import SpeedComparison from './SpeedComparison';
import PyPICard from './PyPICard';
import InteractiveBandDiagram from './InteractiveBandDiagram';
import PitchHeader from './PitchHeader';

export default function BlazePage() {
  return (
    <>
      <BandBackground />
      <PitchHeader />
      <div style={{ minHeight: '100svh', position: 'relative', zIndex: 1 }}>
      
      {/* Hero Section */}
      <section style={{
        position: 'relative',
        zIndex: 1,
        minHeight: '100svh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem',
      }}>
        <h1 style={{
          fontSize: 'clamp(3.5rem, 14vw, 16rem)',
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
          Validated Against MIT Photonic Bands
        </h2>
        <p style={{
          fontSize: '1.1rem',
          color: 'rgba(255, 255, 255, 0.5)',
          textAlign: 'center',
          maxWidth: '100%',
          marginBottom: '3rem',
          overflowWrap: 'anywhere',
        }}>
          Band frequencies compared with MPB under the documented benchmark conditions
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
          Measured performance
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
          overflowWrap: 'anywhere',
        }}>
          Calculate photonic bands and projected operators in your browser.
        </p>
        <InteractiveBandDiagram />
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

