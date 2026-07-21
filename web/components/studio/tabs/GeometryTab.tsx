'use client';

import React, { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { CrystalCanvas } from '../canvas/CrystalCanvas';
import { EpsilonPreview } from '../canvas/EpsilonPreview';

export function GeometryTab() {
  const [epsVisible, setEpsVisible] = useState(true);

  return (
    <div className="studio__tabbody">
      <div className="studio__canvas-wrap">
        <CrystalCanvas />
        <div className="studio__eps-overlay">
          <div className="studio__eps-head">
            <span title="Dielectric map of one unit cell, in lattice coordinates">ε preview</span>
            <button
              type="button"
              className="studio__iconbtn"
              aria-label={epsVisible ? 'Hide dielectric preview' : 'Show dielectric preview'}
              title={epsVisible ? 'Hide' : 'Show'}
              onClick={() => setEpsVisible((v) => !v)}
            >
              {epsVisible ? <EyeOff size={13} /> : <Eye size={13} />}
            </button>
          </div>
          {epsVisible ? <EpsilonPreview size={280} displaySize={140} /> : null}
        </div>
      </div>
    </div>
  );
}
