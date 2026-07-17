'use client';

import { ARCH_MODEL } from '../../lib/archboard/model';
import { KIND_CSS, PRECISION_CSS } from '../../lib/archboard/palette';
import { useBoardStore } from './store';

export default function Tooltip() {
  const hover = useBoardStore((s) => s.hover);
  if (!hover) return null;
  const node = ARCH_MODEL.nodes.find((n) => n.id === hover.id);
  if (!node) return null;

  const chips: { text: string; color: string }[] = [];
  if (node.loc) chips.push({ text: `${(node.loc / 1000).toFixed(1)}k LOC`, color: '#8b949e' });
  if (node.complexity) chips.push({ text: node.complexity, color: KIND_CSS[node.kind] });
  if (node.precision) chips.push({ text: node.precision, color: PRECISION_CSS[node.precision] });

  return (
    <div
      className="archboard__tooltip"
      style={{
        left: Math.min(hover.x + 14, 9999),
        top: hover.y + 14,
      }}
    >
      <div className="archboard__tooltiptitle" style={{ color: KIND_CSS[node.kind] }}>
        {node.label}
      </div>
      <div className="archboard__tooltipshort">{node.short}</div>
      {chips.length > 0 && (
        <div className="archboard__chips">
          {chips.map((chip) => (
            <span key={chip.text} className="archboard__chip" style={{ color: chip.color, borderColor: chip.color }}>
              {chip.text}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
