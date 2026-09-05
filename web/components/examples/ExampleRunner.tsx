'use client';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { getExample } from '../../lib/examples/registry';
import CodeWindow from './CodeWindow';

const Workbench = dynamic(() => import('../workbench/Workbench'), { ssr: false });
export default function ExampleRunner({ slug }: { slug: string }) {
  const example = getExample(slug);
  if (!example) return null;
  return <div className="example-page">
    <section className="example-intro"><Link href="/examples">← Examples</Link><h1>{example.title}</h1><p>{example.description}</p>
      <p>The browser and Python use this same TOML calculation. <Link href={`/workbench/?example=${example.slug}`}>Open in Workbench</Link>.</p>
      <details><summary>Python and TOML files</summary>
        <CodeWindow filename={`${example.slug}.py`} language="python" code={example.python} />
        <CodeWindow filename={`${example.slug}.toml`} language="toml" code={example.source} />
      </details>
    </section>
    <Workbench initialSource={example.source} title={example.title} embedded />
  </div>;
}
