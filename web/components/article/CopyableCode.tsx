'use client';

import { useRef, type ComponentPropsWithoutRef } from 'react';
import { Check, Copy } from 'lucide-react';
import { useCopy } from '@/lib/util/useCopy';

/**
 * A fenced code block that copies itself, from the icon or from anywhere in the
 * block.
 *
 * It wraps every `<pre>` MDX renders, so the affordance arrives with the markup
 * instead of being opted into one block at a time. The text is read back out of
 * the DOM at click time rather than carried as a prop: rehype-pretty-code has
 * already turned the source into a tree of coloured spans by the time this
 * component sees it, and the newlines between lines are real text nodes, so
 * `textContent` reconstructs the listing exactly.
 *
 * The button sits outside the `<pre>` because the `<pre>` scrolls; inside it,
 * the icon would slide away with the code.
 */
export default function CopyableCode({ children, ...props }: ComponentPropsWithoutRef<'pre'>) {
  const pre = useRef<HTMLPreElement>(null);
  const { copied, copy } = useCopy();

  /** `fromButton` marks the unambiguous request, which no selection overrides. */
  const run = (fromButton: boolean) => {
    // A click on the field that ends a drag inside it is someone selecting one
    // argument, not asking for the whole listing; taking the clipboard from
    // them would undo what they were in the middle of doing.
    const selection = window.getSelection();
    if (!fromButton && selection && !selection.isCollapsed && pre.current?.contains(selection.anchorNode)) return;
    // Blank lines are emitted as a single space, which would otherwise be
    // pasted as trailing whitespace.
    const text = (pre.current?.textContent ?? '').replace(/[ \t]+$/gm, '').trim();
    if (text) void copy(text);
  };

  return (
    <div className="code-copy" onClick={() => run(false)}>
      <pre {...props} ref={pre}>{children}</pre>
      <button
        type="button"
        className="code-copy-button"
        data-copied={copied}
        aria-label="Copy this code to the clipboard"
        title={copied ? 'Copied to clipboard' : 'Copy to clipboard'}
        onClick={event => {
          // The wrapper would otherwise copy a second time and restart the flash.
          event.stopPropagation();
          run(true);
        }}
      >
        {copied ? <Check size={15} strokeWidth={2.5} /> : <Copy size={15} />}
        <span aria-live="polite">{copied ? 'Copied' : ''}</span>
      </button>
    </div>
  );
}
