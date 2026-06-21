'use client';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export interface CodeBlockProps {
  code: string;
  language?: string;
  /** Compact card preview vs. full runner view. */
  variant?: 'card' | 'full';
  /** Show line numbers (full view only). */
  showLineNumbers?: boolean;
  /** Override the font size. */
  fontSize?: string;
  /** Override padding. */
  padding?: string;
}

/**
 * Syntax-highlighted code block, matching the markdown python code colouring
 * used elsewhere in the docs (Prism + vscDarkPlus theme).
 *
 * The `blaze-syntax` class is required: a global rule on docs pages strips
 * inline token colours (`code span { color: inherit !important }`), so
 * `app/global.css` re-asserts a scoped vscode-dark theme under `.blaze-syntax`.
 * That same class also hides the scrollbars while keeping the block scrollable.
 */
export default function CodeBlock({
  code,
  language = 'python',
  variant = 'full',
  showLineNumbers = false,
  fontSize,
  padding,
}: CodeBlockProps) {
  return (
    <SyntaxHighlighter
      language={language}
      style={vscDarkPlus}
      showLineNumbers={showLineNumbers}
      wrapLongLines={false}
      className="blaze-syntax"
      customStyle={{
        margin: 0,
        borderRadius: variant === 'card' ? '0' : '0',
        background: '#0d1117',
        fontSize: fontSize ?? (variant === 'card' ? '0.74rem' : '0.82rem'),
        lineHeight: 1.55,
        padding: padding ?? (variant === 'card' ? '14px 16px' : '16px 18px'),
        // Let the surrounding scroll container (CodeWindow's `.subtle-scroll`
        // div) own the vertical scroll + scrollbar. If the highlighter scrolls
        // itself, the `.blaze-syntax` class hides that scrollbar entirely.
        height: 'auto',
        overflow: 'visible',
      }}
      codeTagProps={{
        style: { fontFamily: 'var(--font-mono, ui-monospace, monospace)' },
      }}
    >
      {code}
    </SyntaxHighlighter>
  );
}
