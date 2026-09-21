import Link from 'next/link';
import { posts } from '@/lib/posts.generated';
import { getAssetPath } from '@/lib/paths';
import ApiTomlArt from '@/components/article/ApiTomlArt';
import InstallConsoleArt from '@/components/article/InstallConsoleArt';
import DocumentsFanArt from '@/components/article/DocumentsFanArt';

/**
 * Art for cards that carry a drawing rather than a banner photograph. Each
 * brings the class its own layout needs, because the art decides where the
 * title goes -- above it here, tucked into the opposite corner there.
 */
const ART: Partial<Record<string, { Art: () => React.JSX.Element; className: string }>> = {
  '/configuration': { Art: ApiTomlArt, className: 'post-tile-diagram' },
  '/installation': { Art: InstallConsoleArt, className: 'post-tile-console' },
  '/paper': { Art: DocumentsFanArt, className: 'post-tile-documents' },
};

/** Fixed to UTC so the build and the browser render the same day. */
const DATE = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });

/**
 * The article cards, as one run ordered by `card.order`.
 *
 * That order is a site-wide relevance rank rather than a position inside a
 * section, so the articles a newcomer needs first can lead regardless of which
 * part of the site they belong to. Each card names its own section instead,
 * which is what the labelled groups used to do.
 */
export default function PostGrid() {
  return (
    <div className="posts">
      <div className="post-grid">
        {posts.map(post => {
          const art = ART[post.route];
          return (
          <Link
            href={post.route}
            key={post.route}
            className={`post-tile post-tile-${post.size}${post.showImage ? '' : art ? ` ${art.className}` : ' post-tile-text'}`}
          >
            {post.showImage && (
              <div className="post-image">
                <img src={getAssetPath(post.image)} alt="" loading="lazy" />
              </div>
            )}
            <div className="post-body">
              {post.showTags && <div className="post-tags">{post.tags.map(tag => <span key={tag}>{tag}</span>)}</div>}
              <h3>{post.title}</h3>
              {art && <art.Art />}
              {post.showDescription && <p>{post.description}</p>}
              {post.showDate && 'date' in post && (
                <time dateTime={String(post.date)}>{DATE.format(new Date(String(post.date)))}</time>
              )}
            </div>
          </Link>
          );
        })}
      </div>
    </div>
  );
}
