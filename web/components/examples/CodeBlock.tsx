'use client';

import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import vscDarkPlus from 'react-syntax-highlighter/dist/esm/styles/prism/vsc-dark-plus';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import toml from 'react-syntax-highlighter/dist/esm/languages/prism/toml';

SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('toml', toml);

export type CodeLanguage = 'python' | 'toml';

export interface CodeBlockProps {
  code: string;
  language?: CodeLanguage;
  showLineNumbers?: boolean;
}

/**
 * Runtime-highlighted listing.
 *
 * Fenced code blocks in MDX are highlighted at build time by rehype-pretty-code
 * instead; this component exists for source that is only known at runtime, such
 * as an example's TOML read from the generated catalogue.
 */
export default function CodeBlock({ code, language = 'python', showLineNumbers = false }: CodeBlockProps) {
  return (
    <SyntaxHighlighter
      language={language}
      style={vscDarkPlus}
      showLineNumbers={showLineNumbers}
      wrapLongLines={false}
      className="blaze-syntax"
      customStyle={{
        margin: 0,
        borderRadius: 0,
        background: 'var(--surface-sunken)',
        fontSize: '0.82rem',
        lineHeight: 1.55,
        padding: '16px 18px',
        // The surrounding `.code-window-body` owns the scrollbar; if the
        // highlighter scrolled itself the outer container would never overflow.
        height: 'auto',
        overflow: 'visible',
      }}
      codeTagProps={{ style: { fontFamily: 'var(--font-mono)' } }}
    >
      {code}
    </SyntaxHighlighter>
  );
}
