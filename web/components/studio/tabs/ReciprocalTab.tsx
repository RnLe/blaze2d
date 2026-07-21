'use client';

import React from 'react';
import { useStudioStore } from '../../../lib/studio/store';
import { BrillouinPanel } from '../canvas/BrillouinPanel';

export function ReciprocalTab() {
  const pathMode = useStudioStore((s) => s.config.path.mode);
  const pointCount = useStudioStore((s) => s.config.path.points.length);

  return (
    <div className="studio__tabbody">
      <div className="studio__tabhint">
        {pathMode === 'preset'
          ? 'Preset k-path (read-only). Switch K-path to custom points to edit it here.'
          : pointCount === 0
            ? 'Custom k-path: click anywhere in the zone to add the first point.'
            : 'Click to add points, drag to move them, click a point to remove it.'}
      </div>
      <div className="studio__canvas-wrap">
        <BrillouinPanel />
      </div>
    </div>
  );
}
