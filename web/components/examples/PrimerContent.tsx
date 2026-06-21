import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type { ComponentPropsWithoutRef } from 'react';

// Compact inline styling for primer markdown. The primer is a single short
// paragraph, so we render block elements tightly and keep code/emphasis subtle.
const components = {
  p: (props: ComponentPropsWithoutRef<'p'>) => (
    <p style={{ margin: 0 }} {...props} />
  ),
  code: (props: ComponentPropsWithoutRef<'code'>) => (
    <code
      style={{
        background: '#111827',
        border: '1px solid #1f2937',
        borderRadius: 4,
        padding: '0.05em 0.35em',
        fontSize: '0.85em',
        color: '#93c5fd',
      }}
      {...props}
    />
  ),
  strong: (props: ComponentPropsWithoutRef<'strong'>) => (
    <strong style={{ color: '#f1f5f9', fontWeight: 600 }} {...props} />
  ),
  em: (props: ComponentPropsWithoutRef<'em'>) => (
    <em style={{ color: '#e2e8f0' }} {...props} />
  ),
  a: (props: ComponentPropsWithoutRef<'a'>) => (
    <a style={{ color: '#60a5fa', textDecoration: 'none' }} {...props} />
  ),
};

export default function PrimerContent({ source }: { source: string }) {
  return (
    <MDXRemote
      source={source}
      components={components}
      options={{
        mdxOptions: {
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
      }}
    />
  );
}
