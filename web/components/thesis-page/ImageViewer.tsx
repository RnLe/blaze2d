'use client';

import { useEffect, useId, useRef, useState } from 'react';
import { Expand, ExternalLink, Minus, Plus, X } from 'lucide-react';
import { getAssetPath } from '@/lib/paths';

type Props = {
  src: string;
  title: string;
  alt: string;
  caption: string;
  polarization?: 'TE' | 'TM';
};

/** Original vector figures, with a native modal for focus and keyboard handling. */
export default function ImageViewer({ src, title, alt, caption, polarization }: Props) {
  const [open, setOpen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const dialog = useRef<HTMLDialogElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const id = useId();
  const url = getAssetPath(src);

  useEffect(() => {
    const element = dialog.current;
    if (!open || !element) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    element.showModal();
    return () => {
      element.close();
      document.body.style.overflow = overflow;
    };
  }, [open]);

  function show() {
    setZoom(1);
    setOpen(true);
  }

  function fit() {
    setZoom(1);
    viewport.current?.scrollTo(0, 0);
  }

  return <>
    {polarization ? <button type="button" className="thesis-poster-trigger" onClick={show} aria-haspopup="dialog">
      <span className="thesis-polarization">{polarization}</span>
      <span><strong>Full {polarization} Hamiltonian</strong><small>{polarization === 'TE' ? 'Magnetic field · Hz' : 'Electric field · Ez'}</small><span className="thesis-poster-action">Explore the annotated equation <Expand size={16} aria-hidden="true" /></span></span>
    </button> : <figure className="thesis-figure">
      <button type="button" className="thesis-figure-open" onClick={show} aria-label={`Enlarge figure: ${title}`} aria-haspopup="dialog">
        <img src={url} alt={alt} loading="lazy" />
        <span className="thesis-figure-expand"><Expand size={16} aria-hidden="true" /> Enlarge</span>
      </button>
      <figcaption>{caption}</figcaption>
    </figure>}
    <dialog ref={dialog} className="thesis-dialog" aria-labelledby={`${id}-title`} aria-describedby={`${id}-description`}
      onCancel={event => { event.preventDefault(); setOpen(false); }}
      onClose={() => setOpen(false)}
      onKeyDown={event => {
        if (event.key !== 'Tab') return;
        const controls = event.currentTarget.querySelectorAll<HTMLElement>('button:not([disabled]), a[href], [tabindex="0"]');
        const first = controls[0], last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }}
      onClick={event => {
        if (event.target !== event.currentTarget) return;
        const rect = event.currentTarget.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) setOpen(false);
      }}>
      {open && <>
        <div className="thesis-dialog-header">
          <div><span className="thesis-kicker">Second edition · {polarization ? 'Full operator' : 'Thesis figure'}</span><h2 id={`${id}-title`}>{title}</h2></div>
          <button type="button" className="thesis-icon-button" onClick={() => setOpen(false)} aria-label="Close figure" autoFocus><X aria-hidden="true" /></button>
        </div>
        <div className="thesis-dialog-toolbar">
          <div className="thesis-zoom-controls">
            <button type="button" className="thesis-icon-button" aria-label="Zoom out" disabled={zoom === 1} onClick={() => setZoom(Math.max(1, zoom - 0.5))}><Minus size={16} aria-hidden="true" /></button>
            <output aria-live="polite">{zoom === 1 ? 'Fit' : `${Math.round(zoom * 100)}%`}</output>
            <button type="button" className="thesis-icon-button" aria-label="Zoom in" disabled={zoom === 4} onClick={() => setZoom(Math.min(4, zoom + 0.5))}><Plus size={16} aria-hidden="true" /></button>
            <button type="button" className="thesis-fit-button" onClick={fit}>Fit to view</button>
          </div>
          <a href={url} target="_blank" rel="noreferrer">Open SVG <ExternalLink size={14} aria-hidden="true" /></a>
        </div>
        <div ref={viewport} className={`thesis-dialog-viewport${zoom > 1 ? ' is-zoomed' : ''}`} tabIndex={0} role="region" aria-label="Figure detail, scroll to explore when zoomed">
          <img src={url} alt={alt} style={zoom > 1 ? { width: `${zoom * 100}%` } : undefined} />
        </div>
        <p id={`${id}-description`} className="thesis-dialog-caption">{caption}</p>
      </>}
    </dialog>
  </>;
}
