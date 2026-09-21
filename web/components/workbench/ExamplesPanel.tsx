'use client';
import { useState } from 'react';
import { ArrowLeft, ArrowRight, BookOpen, Check, Download } from 'lucide-react';
import { examples, type Example } from '@/lib/examples/registry';
import { getAssetPath } from '@/lib/paths';
import { download } from '@/lib/compute/export';
import { Choices } from './Choices';

const groups = [
  { name: 'Bands', description: 'Start with a crystal and follow its modes through reciprocal space.' },
  { name: 'Studies', description: 'Compare related crystals in one ordered parameter sweep.' },
  { name: 'Operators', description: 'Inspect the matrix elements behind the bands.' },
] as const;
const notes: Record<Example['slug'], { question: string; inspect: string; try: string; tags: string[] }> = {
  'square-rods': { question: 'How do rods in air shape a band diagram?', inspect: 'Follow eight TM bands along Γ, X, M, and back to Γ. The closing point represents the same wavevector as the start, which makes it a useful consistency check.', try: 'Change the rod radius or its dielectric constant, then compare the new run with the original in History.', tags: ['TM', 'Square', '8 bands'] },
  'triangular-holes': { question: 'What changes when the background is the dielectric?', inspect: 'Here air holes sit in a material with ε = 12. The TE calculation follows Γ, M, K, Γ, the standard path for this triangular lattice.', try: 'Adjust the hole radius and inspect how the separation between neighboring bands changes.', tags: ['TE', 'Triangular', 'Air holes'] },
  'rectangular-cell': { question: 'How does a stretched unit cell change the path?', inspect: 'The direct lattice has a = 2 and b = 1. Its reciprocal-space distances differ from the square case. A 24 × 32 grid also demonstrates independent sampling along each lattice direction.', try: 'Change a or b and compare the geometry with the reciprocal-space path in the Study tab.', tags: ['TM', 'Rectangular', '24 × 32 grid'] },
  'oblique-cell': { question: 'How are distances measured in a skewed lattice?', inspect: 'This cell uses a 70° angle and an explicit sequence of path vertices. Horizontal distances in the band plot use the full reciprocal-space metric, including the skew of the lattice.', try: 'Change the lattice angle. The same fractional path vertices then describe a different Cartesian path.', tags: ['TM', 'Oblique', 'Custom path'] },
  'radius-sweep': { question: 'How do radius and polarization act together?', inspect: 'Five radii and two polarizations produce ten configurations. Polarization is the last sweep axis, so TM and TE are solved consecutively at each radius. The results retain both parameter values.', try: 'Move between configuration samples in Results to compare the two polarizations at a fixed radius.', tags: ['10 configurations', 'Radius × polarization'] },
  'operator-point': { question: 'What can the fields tell us beyond their frequencies?', inspect: 'At one carrier wavevector, Blaze retains two bands starting at band index 2 and includes three upper remote bands. Velocity matrices and inverse-mass data describe the response to changes in wavevector.', try: 'Inspect the array dimensions and residuals in Results. Increase the remote-band count to study sensitivity to the retained spectral window.', tags: ['TE', 'Velocity', 'Inverse mass'] },
  'registry-stencil': { question: 'How do operators vary around a reference configuration?', inspect: 'Two radii and two periodic translations produce four jobs. Each job samples a 3 × 3 grid around the carrier wavevector. The output includes registry derivatives and transported stencil references.', try: 'Compare configuration samples for translations, then stencil samples for nearby wavevectors. Keep these two kinds of variation distinct.', tags: ['4 configurations', '9-point stencil', 'Registry'] },
};

export function ExamplesPanel({ onLoad, disabled, initialSlug, loadedSlug }: {
  onLoad: (example: Example) => void; disabled: boolean; initialSlug?: string; loadedSlug?: string;
}) {
  const [slug, setSlug] = useState<string>(), [sourceTab, setSourceTab] = useState('Overview');
  const example = examples.find(example => example.slug === (slug ?? initialSlug)), detail = example && notes[example.slug];
  if (example && detail) return <div className="wb-examples wb-example-detail">
    <button className="wb-text-button" onClick={() => { setSlug(''); setSourceTab('Overview'); }}><ArrowLeft size={15} />All examples</button>
    <div className="wb-example-heading"><div><span className="wb-eyebrow">{example.category}</span><h1>{example.title}</h1><p>{example.description}</p></div>
      <button className="wb-primary" disabled={disabled} onClick={() => onLoad(example)}><ArrowRight size={16} />Load example</button></div>
    {loadedSlug === example.slug && <p className="wb-loaded"><Check size={14} />Loaded in this workspace. Load again to restore the original configuration.</p>}
    {disabled && <p className="wb-muted">Apply or revert any unfinished edits before loading an example.</p>}
    <div className="wb-example-tags">{detail.tags.map(tag => <span key={tag}>{tag}</span>)}</div>
    <Choices label="Example details" value={sourceTab} compact options={['Overview', 'TOML', 'Python'].map(value => ({ value, label: value }))} onChange={setSourceTab} />
    {sourceTab === 'Overview' ? <>
      <img className="wb-example-art" src={getAssetPath(example.image)} alt={`${example.title}: schematic preview`} width={640} height={310} />
      <section><h2>{detail.question}</h2><p>{detail.inspect}</p></section>
      <section className="wb-example-try"><h2>Try changing one thing</h2><p>{detail.try}</p></section>
      <p className="wb-muted">Loading sets the geometry, task, numerical settings, and sweeps together. The browser uses f64 and a single worker. The Python file uses the same TOML.</p>
    </> : <div className="wb-example-source"><div className="wb-row"><strong>{example.slug}.{sourceTab === 'TOML' ? 'toml' : 'py'}</strong>
      <button onClick={() => download(new Blob([sourceTab === 'TOML' ? example.source : example.python], { type: 'text/plain' }), `${example.slug}.${sourceTab === 'TOML' ? 'toml' : 'py'}`)}><Download size={14} />Download</button></div>
      <pre tabIndex={0}><code>{sourceTab === 'TOML' ? example.source : example.python}</code></pre></div>}
  </div>;
  return <div className="wb-examples">
    <div className="wb-example-heading"><div><span className="wb-eyebrow"><BookOpen size={14} />Example library</span><h1>A starting point for your next calculation</h1>
      <p>Inspect a model, see what it demonstrates, then load it into this workspace.</p></div><span className="wb-library-count">{examples.length} examples</span></div>
    {groups.map(group => <section className="wb-example-group" key={group.name}><div className="wb-row"><h2>{group.name}</h2><span className="wb-muted">{examples.filter(example => example.category === group.name).length} {group.name === 'Studies' ? 'example' : 'examples'}</span></div><p>{group.description}</p>
      <div className="wb-example-grid">{examples.filter(example => example.category === group.name).map(example => <button className="wb-example-card" key={example.slug} onClick={() => { setSlug(example.slug); setSourceTab('Overview'); }}>
        <img src={getAssetPath(example.image)} width={640} height={310} alt="" />
        <span className="wb-example-card-body"><strong>{example.title}<ArrowRight size={15} /></strong><span>{notes[example.slug].question}</span><small>{notes[example.slug].tags.join(' · ')}</small></span>
      </button>)}</div>
    </section>)}
  </div>;
}
