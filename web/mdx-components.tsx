import Link from 'next/link';
import type { MDXComponents } from 'mdx/types';
import { getAssetPath } from './lib/paths';
import CopyableCode from './components/article/CopyableCode';

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    a: ({ href = '', children, ...props }) => href.startsWith('/') && !/\.[a-z0-9]+(?:#.*)?$/i.test(href)
      ? <Link href={href} {...props}>{children}</Link>
      : <a href={href.startsWith('/') ? getAssetPath(href) : href} {...props}>{children}</a>,
    img: ({ src, alt = '', ...props }) => <img src={typeof src === 'string' ? getAssetPath(src) : src} alt={alt} loading="lazy" {...props} />,
    table: props => <div className="article-table" tabIndex={0} role="region" aria-label="Scrollable table"><table {...props} /></div>,
    // Every fenced block copies itself; see CopyableCode. The <CodeWindow> used
    // by <ExampleSource> carries its own copy button and is untouched by this.
    pre: CopyableCode,
    ...components,
  };
}
