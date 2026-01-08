'use client';

import React from 'react';

interface CitationProps {
  id: number;
}

/**
 * Inline citation marker that links to the reference list.
 * Usage: <Citation id={1} />
 */
export function Citation({ id }: CitationProps) {
  return (
    <sup className="citation">
      <a href={`#ref-${id}`} id={`cite-${id}`}>
        [{id}]
      </a>
    </sup>
  );
}

interface ReferenceProps {
  id: number;
  children: React.ReactNode;
}

/**
 * Single reference entry in the reference list.
 * Usage: <Reference id={1}>Author, "Title," Journal, Year.</Reference>
 */
export function Reference({ id, children }: ReferenceProps) {
  return (
    <li id={`ref-${id}`} className="reference-item">
      <a href={`#cite-${id}`} className="reference-backlink" aria-label="Back to citation">
        [{id}]
      </a>
      <span className="reference-text">{children}</span>
    </li>
  );
}

interface ReferencesProps {
  children: React.ReactNode;
}

/**
 * References section wrapper.
 * Usage:
 * <References>
 *   <Reference id={1}>...</Reference>
 *   <Reference id={2}>...</Reference>
 * </References>
 */
export function References({ children }: ReferencesProps) {
  return (
    <section className="references-section">
      <h2>References</h2>
      <ol className="references-list">
        {children}
      </ol>
    </section>
  );
}
