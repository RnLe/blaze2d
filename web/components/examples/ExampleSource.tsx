import { getExample } from '@/lib/examples/registry';
import CodeWindow from './CodeWindow';

export default function ExampleSource({ slug, language }: { slug: string; language: 'python' | 'toml' }) {
  const example = getExample(slug);
  if (!example) throw new Error(`Unknown shared example: ${slug}`);
  return <CodeWindow filename={`${slug}.${language === 'python' ? 'py' : 'toml'}`} language={language} code={language === 'python' ? example.python : example.source} />;
}
