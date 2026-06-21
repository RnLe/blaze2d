'use client';

import type { Example } from '../../lib/examples/registry';
import BlogCard from '../BlogCard';

export interface ExampleCardProps {
  example: Example;
}

// Example cards have no banner image: BlogCard renders an artistic, blurred
// title panel instead (sharpens on hover). The date/tags meta row is hidden;
// only the title and description are shown.
export default function ExampleCard({ example }: ExampleCardProps) {
  return (
    <BlogCard
      href={`/examples/${example.slug}`}
      title={example.title}
      description={example.description}
      showMeta={false}
    />
  );
}
