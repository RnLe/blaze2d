'use client';
import { useState } from 'react';
import { BandComparisonChart } from '../../components/charts';

export default function BandComparisonPlot() {
  const [polarization, setPolarization] = useState<'TE' | 'TM'>('TM');
  return <div style={{ width: '100%', maxWidth: 900 }}>
    <label style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '1rem' }}>Polarization
      <select value={polarization} onChange={event => setPolarization(event.target.value as 'TE' | 'TM')}
        style={{ background: '#17251f', color: '#e2eae5', padding: '.65rem', border: '1px solid #34403b', borderRadius: 7 }}>
        <option value="TM">TM</option><option value="TE">TE</option>
      </select>
    </label>
    <BandComparisonChart width={900} height={480} polarization={polarization} />
    <p style={{ color: 'var(--site-muted)', fontSize: '.9rem', marginTop: '1rem' }}>Separate recorded MPB, Blaze f64, and Blaze mixed-precision datasets are shown.</p>
  </div>;
}
