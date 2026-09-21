import GithubSlugger from 'github-slugger';
import { toString } from 'mdast-util-to-string';

// Shared by the compiler and the indexer so heading links cannot drift.
export default function remarkContent() {
  return tree => {
    const slugger = new GithubSlugger();
    tree.children = tree.children.filter(node => node.type !== 'yaml');
    const walk = node => {
      if (node.type === 'heading') {
        node.data ??= {};
        node.data.hProperties = { ...node.data.hProperties, id: slugger.slug(toString(node)) };
      }
      node.children?.forEach(walk);
    };
    walk(tree);
  };
}
