import Link from 'next/link';
import { notFound } from 'next/navigation';
import ExampleRunner from '../../../components/examples/ExampleRunner';
import { getExample, getExampleSlugs, relocatedExamples } from '../../../lib/examples/registry';

export function generateStaticParams() { return getExampleSlugs().map(slug => ({ slug })); }
export const dynamicParams = false;
export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  return { title: getExample(slug)?.title ?? 'Updated example', robots: relocatedExamples[slug] ? { index: false } : undefined };
}
export default async function ExamplePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  if (relocatedExamples[slug]) return <main className="example-intro"><h1>This example has been updated</h1><p>The configuration API changed in Blaze2D 0.7.</p><Link href={`/examples/${relocatedExamples[slug]}/`}>Open the current example</Link></main>;
  if (!getExample(slug)) notFound();
  return <ExampleRunner slug={slug} />;
}
