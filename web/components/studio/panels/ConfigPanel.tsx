'use client';

import React, { useMemo } from 'react';
import { Sliders } from 'lucide-react';
import { useStudioStore } from '../../../lib/studio/store';
import { SolverSection } from './sections/SolverSection';
import { GeometrySection } from './sections/GeometrySection';
import { GridSection } from './sections/GridSection';
import { PathSection } from './sections/PathSection';
import { SweepsSection } from './sections/SweepsSection';
import { EigensolverSection } from './sections/EigensolverSection';
import { DielectricSection } from './sections/DielectricSection';
import { OutputSection } from './sections/OutputSection';
import { AdvancedSection } from './sections/AdvancedSection';

/** Map a diagnostic path prefix to the accordion section it belongs to. */
export function sectionOf(path: string): string | null {
  if (path.startsWith('geometry')) return 'geometry';
  if (path.startsWith('grid')) return 'grid';
  if (path.startsWith('solver')) return 'solver';
  if (path.startsWith('path')) return 'path';
  if (path.startsWith('sweeps')) return 'sweeps';
  if (path.startsWith('eigensolver')) return 'eigensolver';
  if (path.startsWith('dielectric')) return 'dielectric';
  if (path.startsWith('output')) return 'output';
  if (path.startsWith('operator_data') || path.startsWith('run')) return 'advanced';
  return null;
}

export function ConfigPanel() {
  const diagnostics = useStudioStore((s) => s.toml.diagnostics);

  const sectionErrors = useMemo(() => {
    const set = new Set<string>();
    for (const d of diagnostics) {
      const sec = sectionOf(d.path);
      if (sec) set.add(sec);
    }
    return set;
  }, [diagnostics]);

  return (
    <>
      <div className="studio__panel-title">
        <Sliders size={13} /> Configuration
      </div>
      <div className="studio__panel-scroll subtle-scroll">
        <div className="studio__panel-subhead">Model</div>
        <GeometrySection hasError={sectionErrors.has('geometry')} />
        <GridSection hasError={sectionErrors.has('grid')} />
        <SolverSection />
        <PathSection hasError={sectionErrors.has('path')} />
        <SweepsSection hasError={sectionErrors.has('sweeps')} />
        <div className="studio__panel-subhead">Numerics &amp; output</div>
        <EigensolverSection hasError={sectionErrors.has('eigensolver')} />
        <DielectricSection hasError={sectionErrors.has('dielectric')} />
        <OutputSection hasError={sectionErrors.has('output')} />
        <AdvancedSection />
      </div>
    </>
  );
}
