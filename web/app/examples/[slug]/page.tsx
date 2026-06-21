import { notFound } from 'next/navigation';
import ExampleRunner from '../../../components/examples/ExampleRunner';
import PrimerContent from '../../../components/examples/PrimerContent';
import { getExample, getExampleSlugs } from '../../../lib/examples/registry';

export function generateStaticParams() {
  return getExampleSlugs().map((slug) => ({ slug }));
}

export const dynamicParams = false;

export default async function ExamplePage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const example = getExample(slug);
  if (!example) notFound();
  return (
    <ExampleRunner
      slug={slug}
      primer={example.primer ? <PrimerContent source={example.primer} /> : undefined}
    />
  );
}
