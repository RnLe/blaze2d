type Step = { label: string; space?: 'fourier' | 'real' | 'transform' };

const pipelines: { name: string; operator: string; fft: string; steps: Step[] }[] = [
  {
    name: 'TM',
    operator: 'A = |k + G|², material in the mass B = ε',
    fft: '2 FFTs',
    steps: [
      { label: 'r on the grid', space: 'real' },
      { label: 'FFT', space: 'transform' },
      { label: '÷ (|k + G|² / ⟨ε⟩ + σ)', space: 'fourier' },
      { label: 'inverse FFT', space: 'transform' },
      { label: 'T r', space: 'real' },
    ],
  },
  {
    name: 'TE',
    operator: 'A = −∇·ε⁻¹∇, material inside the operator',
    fft: '6 FFTs',
    steps: [
      { label: 'r on the grid', space: 'real' },
      { label: 'FFT', space: 'transform' },
      { label: 'undo ∇: ÷ (k + G)', space: 'fourier' },
      { label: 'inverse FFTs', space: 'transform' },
      { label: '× ε on the grid', space: 'real' },
      { label: 'FFTs', space: 'transform' },
      { label: 'undo ∇·', space: 'fourier' },
      { label: 'inverse FFT', space: 'transform' },
      { label: 'T r', space: 'real' },
    ],
  },
];

export default function BlazePipelines() {
  return <figure className="pc-pipelines">
    {pipelines.map(pipeline => <div key={pipeline.name} className="pc-pipeline">
      <div className="pc-pipeline-head">
        <strong>{pipeline.name}</strong>
        <span>{pipeline.operator}</span>
        <em>{pipeline.fft}</em>
      </div>
      <ol>
        {pipeline.steps.map((step, index) => <li key={index} data-space={step.space}>{step.label}</li>)}
      </ol>
    </div>)}
    <figcaption>
      <span><i data-space="real" /> real-space grid</span>
      <span><i data-space="fourier" /> Fourier coefficients, one per plane wave k + G</span>
      <span><i data-space="transform" /> fast Fourier transform, O(N log N)</span>
    </figcaption>
  </figure>;
}
