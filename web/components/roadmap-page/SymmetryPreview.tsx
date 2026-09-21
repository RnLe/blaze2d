// Interior samples avoid identifying opposite Brillouin-zone boundary points.
// The square point group maps (x,y) to a representative max(|x|,|y|), min(|x|,|y|).
const samples = Array.from({ length: 81 }, (_, i) => {
  const x = i % 9 - 4, y = Math.floor(i / 9) - 4;
  return { x, y, representative: x >= 0 && y >= 0 && y <= x };
});

function Zone({ reduced = false }: { reduced?: boolean }) {
  const points = reduced ? samples.filter(point => point.representative) : samples;
  return <div className="roadmap-symmetry-panel">
    <svg viewBox="30 5 225 200" role="img" aria-label={reduced ? '15 representatives of 81 sampled wavevectors' : '81 sampled wavevectors in the square Brillouin zone'}>
      <desc>For a square-symmetric crystal, rotations and reflections group this illustrative nine by nine interior grid into fifteen sets. The triangular wedge contains one representative from each set.</desc>
      <rect className="roadmap-zone" x="47" y="19" width="176" height="176" />
      {reduced && <path className="roadmap-wedge" d="M135,107 L223,107 L223,19 Z" />}
      <path className="roadmap-zone-axes" d="M39,107 H231 M135,11 V203" />
      {points.map(({ x, y }) => <circle key={`${x},${y}`} cx={135 + x * 18} cy={107 - y * 18}
        r={reduced ? 3.8 : 3.1}
        className={`roadmap-kpoint${reduced ? ' is-representative' : ''}`} />)}
      <text x="125" y="126">Γ</text><text x="235" y="113">X</text><text x="233" y="23">M</text>
    </svg>
    <div className="roadmap-symmetry-count">
      <strong>{points.length}</strong><span>{reduced ? 'representatives' : 'k-points'}</span>
    </div>
  </div>;
}

export default function SymmetryPreview() {
  return <figure className="roadmap-banner roadmap-symmetry" data-visual-for="symmetries">
    <Zone />
    <div className="roadmap-symmetry-operation">
      <strong>C₄ᵥ symmetry</strong>
      <svg viewBox="0 0 140 24" aria-hidden="true"><path d="M4 12H134 M125 4l10 8-10 8" /></svg>
      <span>Rotations and reflections</span>
    </div>
    <Zone reduced />
  </figure>;
}
