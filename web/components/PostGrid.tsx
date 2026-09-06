import Link from 'next/link';
import { posts } from '../lib/posts.generated';
import { getAssetPath } from '../lib/paths';

export default function PostGrid() {
  return <div className="posts">{['Use Blaze', 'Research'].map(category => <section className="post-group" key={category} aria-label={category}>
    <h2>{category}</h2><div className="post-grid">{posts.filter(post => post.category === category).map(post => <Link href={post.route} key={post.route}
      className={`post-tile post-tile-${post.size}${post.showImage ? '' : ' post-tile-text'}`}>
      {post.showImage && <div className="post-image"><img src={getAssetPath(post.image)} alt="" loading="lazy" /></div>}
      <div className="post-body">
        {post.showTags && <div className="post-tags">{post.tags.map(tag => <span key={tag}>{tag}</span>)}</div>}
        {post.showDate && 'date' in post && <time dateTime={String(post.date)}>{String(post.date)}</time>}
        <h3>{post.title}</h3>{post.showDescription && <p>{post.description}</p>}
        <span className="post-open" aria-hidden="true">Read <span>↗</span></span>
      </div>
    </Link>)}</div>
  </section>)}</div>;
}
