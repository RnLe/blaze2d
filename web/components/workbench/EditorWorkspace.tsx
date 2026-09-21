'use client';
import { useRef, useState, type ReactNode } from 'react';
import { BookOpen, Braces, ChartNoAxesCombined, Grid2X2, SlidersHorizontal, X } from 'lucide-react';

export const editorViews = ['Geometry', 'Study', 'Results', 'TOML', 'Examples'] as const;
export type EditorViewName = typeof editorViews[number];
export type EditorGroup = 'primary' | 'secondary';
const icons = { Geometry: Grid2X2, Study: SlidersHorizontal, Results: ChartNoAxesCombined, TOML: Braces, Examples: BookOpen };

export function EditorWorkspace({ primary, secondary, focused, onActivate, onFocus, onCloseSplit, children, pending }: {
  primary: EditorViewName; secondary: EditorViewName | null; focused: EditorGroup;
  onActivate: (view: EditorViewName, group: EditorGroup) => void; onFocus: (group: EditorGroup) => void;
  onCloseSplit: () => void; children: Record<EditorViewName, ReactNode>; pending: boolean;
}) {
  const host = useRef<HTMLDivElement>(null), [ratio, setRatio] = useState(50);
  function tabs(group: EditorGroup, selected: EditorViewName) {
    return <div className={`wb-editor-tabs wb-${group}-tabs`} data-focused={focused === group}>
      <div role="tablist" aria-label={group === 'primary' ? 'Editor tabs' : 'Split editor tabs'}>{editorViews.map((name, index) => {
        const Icon = icons[name];
        return <button role="tab" id={`wb-${group}-tab-${name}`} aria-controls={`wb-panel-${name}`} aria-selected={selected === name}
          tabIndex={selected === name ? 0 : -1} key={name} onClick={() => onActivate(name, group)} onKeyDown={event => {
            const next = event.key === 'ArrowRight' ? (index + 1) % editorViews.length : event.key === 'ArrowLeft' ? (index + editorViews.length - 1) % editorViews.length : event.key === 'Home' ? 0 : event.key === 'End' ? editorViews.length - 1 : -1;
            if (next >= 0) { event.preventDefault(); onActivate(editorViews[next], group); document.getElementById(`wb-${group}-tab-${editorViews[next]}`)?.focus(); }
          }}><Icon size={15} /><span>{name}</span>{name === 'TOML' && pending && <span className="wb-unsaved" aria-label="Unapplied changes" />}</button>;
      })}</div>
      {group === 'secondary' && <button className="wb-close-split" aria-label="Close split editor" onClick={onCloseSplit}><X size={15} /></button>}
    </div>;
  }
  return <div ref={host} className={`wb-editors${secondary ? ' wb-editors-split' : ''}`}
    style={secondary ? { gridTemplateColumns: `minmax(0, ${ratio}fr) 5px minmax(0, ${100 - ratio}fr)` } : undefined}>
    {tabs('primary', primary)}{secondary && tabs('secondary', secondary)}
    {secondary && <div className="wb-splitter" role="separator" aria-label="Editor split" aria-orientation="vertical" tabIndex={0}
      aria-valuemin={30} aria-valuemax={70} aria-valuenow={Math.round(ratio)} onKeyDown={event => {
        if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); setRatio(value => Math.max(30, Math.min(70, value + (event.key === 'ArrowLeft' ? -5 : 5)))); }
      }} onPointerDown={event => { event.currentTarget.setPointerCapture(event.pointerId); }}
      onPointerMove={event => { if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
        const bounds = host.current!.getBoundingClientRect(); setRatio(Math.max(30, Math.min(70, (event.clientX - bounds.left) / bounds.width * 100)));
      }} onPointerUp={event => event.currentTarget.releasePointerCapture(event.pointerId)} />}
    {editorViews.map(name => {
      const group = secondary === name ? 'secondary' : 'primary';
      return <section key={name} id={`wb-panel-${name}`} className={`wb-editor-view wb-${group}-view`} role="tabpanel" tabIndex={0}
        aria-labelledby={`wb-${group}-tab-${name}`} hidden={primary !== name && secondary !== name}
        onPointerDown={() => onFocus(group)} onFocus={() => onFocus(group)}>{children[name]}</section>;
    })}
  </div>;
}
