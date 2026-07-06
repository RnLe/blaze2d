'use client';

import { ARCH_MODEL } from '../../lib/archboard/model';
import { KIND_CSS, KIND_LABELS, PRECISION_CSS } from '../../lib/archboard/palette';
import { getAssetPath } from '../../lib/paths';
import { useBoardStore } from './store';

const GITHUB_BASE = 'https://github.com/RnLe/blaze2d/blob/main/';

export default function DetailPanel({ onClose }: { onClose: () => void }) {
  const selected = useBoardStore((s) => s.selected);
  const node = selected ? ARCH_MODEL.nodes.find((n) => n.id === selected) : null;

  return (
    <div className={`archboard__panel${node ? ' archboard__panel--open' : ''}`}>
      {node && (
        <>
          <button className="archboard__panelclose" onClick={onClose} aria-label="Close details">
            ×
          </button>
          <div className="archboard__chips" style={{ marginTop: 0 }}>
            <span
              className="archboard__chip"
              style={{ color: KIND_CSS[node.kind], borderColor: KIND_CSS[node.kind] }}
            >
              {KIND_LABELS[node.kind]}
            </span>
            {node.precision && (
              <span
                className="archboard__chip"
                style={{ color: PRECISION_CSS[node.precision], borderColor: PRECISION_CSS[node.precision] }}
              >
                {node.precision}
              </span>
            )}
            {node.badge && (
              <span className="archboard__chip" style={{ color: '#fff', borderColor: 'rgba(255,255,255,0.4)' }}>
                {node.badge}
              </span>
            )}
          </div>
          <div className="archboard__paneltitle">{node.label}</div>
          <div className="archboard__paneldesc">{renderInlineCode(node.description ?? node.short)}</div>

          <table className="archboard__metrics">
            <tbody>
              {node.complexity && (
                <tr>
                  <td>cost</td>
                  <td>{node.complexity}</td>
                </tr>
              )}
              {node.loc && (
                <tr>
                  <td>size</td>
                  <td>{node.loc.toLocaleString()} LOC</td>
                </tr>
              )}
              {node.parallel && (
                <tr>
                  <td>parallelism</td>
                  <td>{parallelLabel(node.parallel)}</td>
                </tr>
              )}
              {node.metrics?.map(([key, value]) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="archboard__panellinks">
            {node.file && (
              <a className="archboard__panellink" href={GITHUB_BASE + node.file} target="_blank" rel="noreferrer">
                {node.file} ↗
              </a>
            )}
            {node.subpage && (
              <a className="archboard__panellink" href={getAssetPath(node.subpage) + '/'}>
                Deep dive: {node.subpage.split('/').pop()} →
              </a>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function parallelLabel(mode: string): string {
  switch (mode) {
    case 'rayon-jobs':
      return 'rayon, data-parallel over jobs';
    case 'serial-k':
      return 'sequential by design (warm-start chain)';
    case 'gemm-seq-by-design':
      return 'sequential by design (Par::Seq)';
    case 'browser-worker':
      return 'Web Worker (off the main thread)';
    case 'py-worker-thread':
      return 'Rust worker thread, GIL released';
    default:
      return mode;
  }
}

/** Renders `code` spans inside plain description text. */
function renderInlineCode(text: string) {
  const parts = text.split(/(`[^`]+`)/g);
  return parts.map((part, i) =>
    part.startsWith('`') && part.endsWith('`') ? <code key={i}>{part.slice(1, -1)}</code> : <span key={i}>{part}</span>
  );
}
