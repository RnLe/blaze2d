import BandBackground from './BandBackground';
import BandComparisonPlot from './BandComparisonPlot';
import SpeedComparison from './SpeedComparison';
import InstallCards from './InstallCard';
import InteractiveBandDiagram from './InteractiveBandDiagram';
import PitchHeader from './PitchHeader';

export default function PitchPage() {
  return (
    <>
      <BandBackground />
      <PitchHeader />
      <div className="pitch-sections">
        <section className="pitch-section pitch-hero">
          <h1>BLAZE 2D</h1>
          <p>A fast Rust-based 2D Maxwell solver for photonic band structures</p>
          <div className="pitch-scroll-hint">↓ Scroll to explore ↓</div>
        </section>

        <section className="pitch-section">
          <h2>Validated against MIT Photonic Bands</h2>
          <p>Cross-validated with the reference implementation for photonic band structure computation</p>
          <BandComparisonPlot />
        </section>

        <section className="pitch-section">
          <h2>Blazing fast</h2>
          <SpeedComparison />
        </section>

        <section className="pitch-section">
          <h2>Try it yourself</h2>
          <p>Compute photonic band structures directly in your browser. No installation required.</p>
          <InteractiveBandDiagram />
        </section>

        <section className="pitch-section">
          <InstallCards />
        </section>
      </div>
    </>
  );
}
