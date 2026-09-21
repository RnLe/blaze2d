import { readdir, readFile, writeFile, access, mkdir, rm } from 'node:fs/promises';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMdx from 'remark-mdx';
import remarkMath from 'remark-math';
import { toString } from 'mdast-util-to-string';
import remarkContent from './content.mjs';
import { describeDocument } from './documents.mjs';

const directory = new URL('../content/', import.meta.url);
const publicDirectory = new URL('../public/', import.meta.url);
const processor = unified().use(remarkParse).use(remarkMdx).use(remarkMath).use(remarkContent);
const articles = [], posts = [], navigation = [], search = [];
// PDFs an article links to, measured once and keyed by their public path.
const documents = new Map();
const cardKeys = new Set(['title', 'category', 'order', 'size', 'description', 'image', 'date', 'tags', 'showDescription', 'showDate', 'showTags', 'showImage']);
for (const file of (await readdir(directory)).sort()) {
  if (!/\.mdx?$/.test(file)) continue;
  const { data, content } = matter(await readFile(new URL(file, directory), 'utf8'));
  const fail = message => { throw new Error(`${file}: ${message}`); };
  if (data.draft === true) continue;
  const slug = file.replace(/\.mdx?$/, '').replace(/^index$/, '');
  if (articles.some(article => article.slug === slug)) fail('Duplicate route');
  const tree = await processor.run(processor.parse(content));
  const headings = [], text = [];
  function collect(node) {
    if (node.type === 'heading') headings.push({ text: toString(node), id: node.data.hProperties.id, depth: node.depth });
    if (['paragraph', 'heading', 'code'].includes(node.type)) text.push(toString(node));
    else node.children?.forEach(collect);
  }
  collect(tree);
  const title = data.title ?? headings.find(heading => heading.depth === 1)?.text;
  if (typeof title !== 'string' || !title.trim()) fail('A title is required');
  if (data.layout && !['article', 'wide', 'home'].includes(data.layout)) fail('Unknown layout');
  if (data.redirect && (typeof data.redirect !== 'string' || !data.redirect.startsWith('/') || data.redirect.startsWith('//'))) fail('redirect must be an internal destination');
  const route = data.redirect ?? '/' + slug;
  const description = data.description ?? data.card?.description ?? text.find(line => line !== title)?.slice(0, 180) ?? '';
  // Pages with H1 sections use H1/H2 navigation after the opening page title.
  const hasTopLevelSections = headings.filter(heading => heading.depth === 1).length > 1;
  const toc = hasTopLevelSections
    ? headings.slice(headings.findIndex(heading => heading.depth === 1) + 1).filter(heading => heading.depth <= 2)
    : headings.filter(heading => heading.depth === 2 || heading.depth === 3);
  articles.push({ slug, file, title, description, layout: data.layout ?? 'article', robots: data.robots, toc, redirect: data.redirect });
  if (data.robots?.index !== false) search.push({ route, title, description, headings: toc, text: text.join('\n') });
  const card = data.card;
  if (card) {
    for (const key of Object.keys(card)) if (!cardKeys.has(key)) fail(`Unknown card field ${key}`);
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
    posts.push({ ...card, route });
  }
  let document;
  if (data.document !== undefined) {
    if (typeof data.document !== 'string' || !data.document.startsWith('/')) fail('document must be a public asset path');
    try { document = documents.get(data.document) ?? describeDocument(publicDirectory, data.document); }
    catch (error) { fail(error.message); }
    documents.set(data.document, document);
  }
  // `nav.order` places a page in its sidebar group; `card.order` ranks a card
  // against every other card. They are different rankings, so a page that
  // wants both states both.
  const nav = data.nav;
  if (nav) {
    if (!['Use Blaze', 'Research', 'Theory', 'Project'].includes(nav.group) || !Number.isInteger(nav.order)) fail('Navigation needs a group and integer order');
    if (nav.document !== undefined && typeof nav.document !== 'boolean') fail('nav.document must be boolean');
    if (nav.document && !document) fail('nav.document needs a document to describe');
    // Measuring a document and advertising it in navigation are separate
    // choices: an article can offer a download without the sidebar repeating it.
    navigation.push({ route, title: nav.title ?? title, group: nav.group, order: nav.order, document: nav.document ? document : undefined });
  }
}
if (new Set(posts.map(post => post.order)).size !== posts.length) throw new Error('Duplicate card order');
for (const group of ['Use Blaze', 'Research', 'Theory', 'Project']) {
  const entries = navigation.filter(item => item.group === group);
  if (new Set(entries.map(item => item.order)).size !== entries.length) throw new Error(`Duplicate order in ${group}`);
}
// One run, most relevant first; each card names its own section.
posts.sort((a, b) => a.order - b.order);
navigation.sort((a, b) => a.order - b.order);
const generated = '/* Generated from content metadata. Run pnpm generate:content to update. */\n';
await writeFile(new URL('../lib/posts.generated.ts', import.meta.url), generated + 'export const posts = ' + JSON.stringify(posts, null, 2) + ' as const;\n');
await writeFile(new URL('../lib/navigation.generated.ts', import.meta.url), generated +
  "import type { DocumentInfo } from './documents';\n\n" +
  'export interface NavigationItem {\n' +
  '  route: string;\n' +
  '  title: string;\n' +
  "  group: 'Use Blaze' | 'Research' | 'Theory' | 'Project';\n" +
  '  order: number;\n' +
  '  /** Set when the article is about a PDF, so navigation can advertise it. */\n' +
  '  document?: DocumentInfo;\n' +
  '}\n\n' +
  'export const navigation: NavigationItem[] = ' + JSON.stringify(navigation, null, 2) + ';\n');
await writeFile(new URL('../lib/search.generated.json', import.meta.url), JSON.stringify(search));
await writeFile(new URL('../lib/documents.generated.ts', import.meta.url), generated +
  "import type { DocumentInfo } from './documents';\n\n" +
  'export const documents: Record<string, DocumentInfo> = ' + JSON.stringify(Object.fromEntries(documents), null, 2) + ';\n');
const routes = new URL('../app/(docs)/(content)/', import.meta.url);
await rm(routes, { recursive: true, force: true });
for (const { slug, file, title, description, layout, robots, toc, redirect } of articles) {
  const folder = new URL(slug ? `${slug}/` : './', routes);
  await mkdir(folder, { recursive: true });
  if (redirect) {
    await writeFile(new URL('page.tsx', folder), generated +
      `import { WorkbenchRedirect } from '@/components/examples/WorkbenchRedirect';\n` +
      `export const metadata = ${JSON.stringify({ title, description, robots: { index: false } })};\n` +
      `export default function Page() { return <WorkbenchRedirect href=${JSON.stringify(redirect)} />; }\n`);
    continue;
  }
  await writeFile(new URL('page.tsx', folder), generated +
    `import Content from '@/content/${file}';\nimport { ArticleFrame } from '@/components/site/ArticleFrame';\n` +
    `export const metadata = ${JSON.stringify({ title, description, robots })};\n` +
    `export default function Page() { return <ArticleFrame layout=${JSON.stringify(layout)} toc={${JSON.stringify(toc)}}><Content /></ArticleFrame>; }\n`);
}
console.log(`Indexed ${articles.length} articles, ${posts.length} cards, and ${documents.size} documents.`);
