import { RefreshCcw } from 'lucide-react';

const steps = [
  { name: 'Residual', formula: 'R = AX − XΛ', text: 'How far each approximate band is from an eigenvector; Λ holds the Rayleigh quotients.' },
  { name: 'Precondition', formula: 'W = T R', text: 'Turn every residual into a proposed correction.', highlight: true },
  { name: 'Trial subspace', formula: 'S = span{X, W, P}', text: 'Current bands, corrections W and the previous update P.' },
  { name: 'Rayleigh–Ritz', formula: 'X ← best vectors in S', text: 'Keep the combinations with the smallest Rayleigh quotients.' },
];

export default function LobpcgCycle() {
  return <figure className="pc-cycle">
    <ol>
      {steps.map((step, index) => <li key={step.name} data-highlight={step.highlight || undefined}>
        <span className="pc-cycle-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
        <strong>{step.name}</strong>
        <code>{step.formula}</code>
        <span>{step.text}</span>
      </li>)}
    </ol>
    <figcaption><RefreshCcw size={15} aria-hidden="true" /> One LOBPCG iteration, repeated at every wavevector until the bands converge. Only step 2 decides which new directions enter the subspace.</figcaption>
  </figure>;
}
