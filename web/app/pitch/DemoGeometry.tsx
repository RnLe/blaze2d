'use client';
import { useMemo } from 'react';
import { epsilonColor } from '@/lib/theme';
import { analyticGeometry, PREVIEW_PERIODS, type DemoSettings } from './demo-model';

export default function DemoGeometry({ settings }: { settings: DemoSettings }) {
  const { crystal, ratio, radius } = settings;
  const geometry = useMemo(() => analyticGeometry({ crystal, ratio, radius }), [crystal, ratio, radius]);
  const { span, circles } = geometry;
  return <figure className="pitch-demo-preview">
    <svg viewBox={`${-span / 2} ${-span / 2} ${span} ${span}`} role="img"
      aria-label="Analytic crystal geometry, spanning seven times the shortest lattice constant"
      data-periods={PREVIEW_PERIODS} data-world-span={span} data-radius={radius}>
      <rect x={-span / 2} y={-span / 2} width={span} height={span} fill={epsilonColor((settings.background - 1) / 12)} />
      <g fill={epsilonColor((settings.inclusion - 1) / 12)}>
        {circles.map(circle => <circle key={circle.key} cx={circle.x} cy={-circle.y} r={radius} />)}
      </g>
    </svg>
  </figure>;
}
