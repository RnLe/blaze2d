'use client';

import { KIND_CSS, KIND_LABELS, PRECISION_CSS, PACKET_PRECISION_COLORS, PACKET_COLORS, cssColor } from '../../lib/archboard/palette';
import type { NodeKind } from '../../lib/archboard/types';
import { useBoardStore } from './store';

const KIND_ORDER: NodeKind[] = ['compute', 'movement', 'memory', 'interface', 'control'];

export default function Legend() {
  const view = useBoardStore((s) => s.view);

  if (view === 'precision') {
    return (
      <div className="archboard__legend">
        <div className="archboard__legendrow" style={{ fontWeight: 700, color: '#fff' }}>
          Precision overlay
        </div>
        <div className="archboard__legendrow">
          <span className="archboard__swatch" style={{ borderColor: PRECISION_CSS.f32 }} /> f32 storage path
        </div>
        <div className="archboard__legendrow">
          <span className="archboard__swatch" style={{ borderColor: PRECISION_CSS.f64 }} /> f64 always
        </div>
        <div className="archboard__legendrow">
          <span
            className="archboard__swatch"
            style={{
              borderColor: PRECISION_CSS.f64,
              background: `linear-gradient(90deg, ${PRECISION_CSS.f32}44 50%, ${PRECISION_CSS.f64}44 50%)`,
            }}
          />
          mixed (casts inside)
        </div>
        <div className="archboard__legendrow">
          <span
            className="archboard__swatch"
            style={{ borderColor: '#fff', transform: 'rotate(45deg)', borderRadius: 1, width: 10, height: 10 }}
          />
          f32 ↔ f64 boundary
        </div>
      </div>
    );
  }

  if (view === 'run') {
    return (
      <div className="archboard__legend">
        <div className="archboard__legendrow" style={{ fontWeight: 700, color: '#fff' }}>
          Packets by payload
        </div>
        <div className="archboard__legendrow">
          <span
            className="archboard__swatch--dot"
            style={{ background: cssColor(PACKET_PRECISION_COLORS.f32) }}
          />
          f32 payload (storage)
        </div>
        <div className="archboard__legendrow">
          <span
            className="archboard__swatch--dot"
            style={{ background: cssColor(PACKET_PRECISION_COLORS.f64) }}
          />
          f64 payload (accumulate, results)
        </div>
        <div className="archboard__legendrow">
          <span className="archboard__swatch--dot" style={{ background: cssColor(PACKET_COLORS.control) }} />
          control &amp; config
        </div>
      </div>
    );
  }

  return (
    <div className="archboard__legend">
      {KIND_ORDER.map((kind) => (
        <div key={kind} className="archboard__legendrow">
          <span className="archboard__swatch" style={{ borderColor: KIND_CSS[kind] }} />
          {KIND_LABELS[kind]}
        </div>
      ))}
    </div>
  );
}
