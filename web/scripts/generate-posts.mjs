import { readdir, readFile, writeFile, access } from 'node:fs/promises';
import matter from 'gray-matter';

const content = new URL('../content/', import.meta.url), posts = [];
const keys = new Set(['title', 'category', 'order', 'size', 'description', 'image', 'date', 'tags', 'showDescription', 'showDate', 'showTags', 'showImage']);
for (const file of await readdir(content)) {
  if (!file.endsWith('.mdx')) continue;
  const { data } = matter(await readFile(new URL(file, content), 'utf8'));
  if (!data.card) continue;
  const card = data.card, fail = message => { throw new Error(`${file}: ${message}`); };
  for (const key of Object.keys(card)) if (!keys.has(key)) fail(`Unknown card field ${key}`);
  if (!['Use Blaze', 'Research'].includes(card.category) || !['1x1', '2x1', '1x2', '2x2'].includes(card.size)) fail('Invalid category or tile size');
  if (typeof card.title !== 'string' || !card.title.trim() || !Number.isInteger(card.order)) fail('Card needs a title and integer order');
  for (const key of ['showDescription', 'showDate', 'showTags', 'showImage']) if (typeof card[key] !== 'boolean') fail(`${key} must be boolean`);
  if (card.showDescription && typeof card.description !== 'string') fail('Missing description');
  if (card.showDate && (typeof card.date !== 'string' || Number.isNaN(Date.parse(card.date)))) fail('Missing or invalid date');
  if (!Array.isArray(card.tags) || card.tags.some(tag => typeof tag !== 'string')) fail('Tags must be text');
  if (card.showImage) {
    if (typeof card.image !== 'string' || !card.image.startsWith('/')) fail('Image must be a public asset');
    await access(new URL('../public' + card.image, import.meta.url));
  }
  const route = '/' + file.slice(0, -4);
  await access(new URL(file, content));
  posts.push({ ...card, route });
}
for (const category of ['Use Blaze', 'Research']) {
  const orders = posts.filter(post => post.category === category).map(post => post.order);
  if (new Set(orders).size !== orders.length) throw new Error(`Duplicate card order in ${category}`);
}
posts.sort((a, b) => ['Use Blaze', 'Research'].indexOf(a.category) - ['Use Blaze', 'Research'].indexOf(b.category) || a.order - b.order);
await writeFile(new URL('../lib/posts.generated.ts', import.meta.url),
  '/* Generated from per-post card metadata. Run pnpm generate:posts to update. */\nexport const posts = ' + JSON.stringify(posts, null, 2) + ' as const;\n');
