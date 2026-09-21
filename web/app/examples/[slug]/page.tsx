import { notFound } from 'next/navigation';
import { WorkbenchRedirect } from '@/components/examples/WorkbenchRedirect';
import { getExample, getExampleSlugs, relocatedExamples } from '@/lib/examples/registry';

export function generateStaticParams() { return getExampleSlugs().map(slug => ({ slug })); }
export const dynamicParams = false;
export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  return { title: getExample(slug)?.title ?? 'Updated example', robots: { index: false } };
}
export default async function ExamplePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const current = relocatedExamples[slug] ?? slug;
  if (!getExample(current)) notFound();
  return <WorkbenchRedirect href={`/workbench?view=examples&inspect=${current}`} />;
}
