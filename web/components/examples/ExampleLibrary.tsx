import Link from 'next/link';
import { examples } from '../../lib/examples/registry';

export default function ExampleLibrary() {
  return <div className="example-library">{['Bands', 'Studies', 'Operators'].map(category => <section key={category}>
    <h2>{category}</h2><div className="example-links">{examples.filter(example => example.category === category).map(example =>
      <Link href={`/examples/${example.slug}/`} key={example.slug}><h3>{example.title}</h3><p>{example.description}</p><span>Open example →</span></Link>)}</div>
  </section>)}</div>;
}
