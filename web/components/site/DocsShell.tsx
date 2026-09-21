'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, FlaskConical, Menu, Search, X } from 'lucide-react';
import { navigation } from '@/lib/navigation.generated';
import { getAssetPath } from '@/lib/paths';
import { SiteFooter } from '@/components/site/SiteFooter';
import { GitHubIcon, PdfIcon } from '@/components/site/icons';
import { formatDocument } from '@/lib/documents';

type SearchEntry = { route: string; title: string; description: string; text: string; headings: { text: string; id: string }[] };
export function DocsShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [shortcut, setShortcut] = useState('Ctrl K');
  useEffect(() => { if (/Mac|iPhone|iPad/.test(navigator.platform)) setShortcut('⌘ K'); }, []);
  const menu = useRef<HTMLDialogElement>(null), search = useRef<HTMLDialogElement>(null), input = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState(''), [index, setIndex] = useState<SearchEntry[]>(), [searchError, setSearchError] = useState(false);
  useEffect(() => { menu.current?.close(); search.current?.close(); }, [pathname]);
  // Hovering a card on the home page lights up the sidebar entry it leads to.
  // The two live in different subtrees, so no selector can relate them; the
  // shell owns both and listens for the card instead.
  const [linked, setLinked] = useState('');
  useEffect(() => {
    const routeOf = (target: EventTarget | null) => {
      const tile = target instanceof Element ? target.closest<HTMLAnchorElement>('a.post-tile') : null;
      return tile ? new URL(tile.href).pathname.replace(/\/$/, '') : '';
    };
    const enter = (event: PointerEvent) => { const route = routeOf(event.target); if (route) setLinked(route); };
    const leave = (event: PointerEvent) => { if (routeOf(event.target)) setLinked(''); };
    document.addEventListener('pointerover', enter);
    document.addEventListener('pointerout', leave);
    return () => { document.removeEventListener('pointerover', enter); document.removeEventListener('pointerout', leave); };
  }, []);
  function openSearch() {
    menu.current?.close(); search.current?.showModal(); setQuery(''); setSearchError(false); input.current?.focus();
    if (!index) void import('@/lib/search.generated.json').then(module => setIndex(module.default)).catch(() => setSearchError(true));
  }
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); openSearch(); }
    };
    document.addEventListener('keydown', handler); return () => document.removeEventListener('keydown', handler);
  });
  const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const results = index?.map(entry => {
    const title = entry.title.toLowerCase(), body = entry.text.toLowerCase();
    return { ...entry, score: terms.every(term => title.includes(term) || body.includes(term)) ? terms.reduce((sum, term) => sum + (title.includes(term) ? 10 : 1), 0) : -1 };
  }).filter(entry => entry.score >= 0).sort((a, b) => b.score - a.score).slice(0, 10) ?? [];
  function links() {
    return <>
      <Link href="/workbench" className="site-workbench-link"><FlaskConical size={18} />Workbench<ArrowUpRight size={16} /></Link>
      {['Use Blaze', 'Research', 'Theory', 'Project'].map(group => {
        const items = navigation.filter(item => item.group === group);
        return <div className="site-nav-group" key={group}>
        <p className="site-nav-heading">{group}</p>
        <div className="site-nav-items">
          {items.map(item => <Link href={item.route} key={item.route}
            aria-current={pathname.replace(/\/$/, '') === item.route ? 'page' : undefined}
            data-linked={item.route === linked ? '' : undefined}>
            <span>{item.title}</span>
            {item.document && <small>(<PdfIcon size={12} />{formatDocument(item.document)})</small>}
          </Link>)}
          {group === 'Project' && <Link href="/pitch"><span>Pitch</span><ArrowUpRight size={14} /></Link>}
        </div>
      </div>;
      })}
    </>;
  }
  return <div className="docs-shell">
    <a href="#main-content" className="skip-link">Skip to content</a>
    <header className="site-header">
      <Link href="/" className="site-brand"><img src={getAssetPath('/icons/blaze_bw.svg')} alt="" width={28} height={28} />Blaze2D</Link>
      <div className="site-header-actions"><button className="site-search-button" onClick={openSearch}><Search size={17} /><span>Search</span><kbd>{shortcut}</kbd></button>
        <a className="site-source-link" href="https://github.com/RnLe/blaze2d"><GitHubIcon size={17} />Source<ArrowUpRight size={15} /></a>
        <button className="site-menu-button" aria-label="Open navigation" onClick={() => menu.current?.showModal()}><Menu size={21} /></button>
      </div>
    </header>
    <div className="site-grid"><aside className="site-sidebar"><nav aria-label="Main navigation">{links()}</nav></aside>
      <div className="site-content"><main id="main-content" tabIndex={-1}>{children}</main><footer className="site-footer"><SiteFooter /></footer></div>
    </div>
    <dialog ref={menu} className="site-mobile-menu" aria-label="Navigation"><div className="site-dialog-heading"><strong>Blaze2D</strong><button aria-label="Close navigation" onClick={() => menu.current?.close()}><X size={20} /></button></div><nav aria-label="Mobile navigation">{links()}</nav></dialog>
    <dialog ref={search} className="site-search-dialog" aria-label="Search documentation" onClick={event => { if (event.target === search.current) search.current.close(); }}>
      <div className="site-search-inner"><div className="site-search-input"><Search size={20} /><input ref={input} type="search" aria-label="Search documentation" placeholder="Search articles, examples, and API…" value={query} onChange={event => setQuery(event.target.value)}
        onKeyDown={event => { if (event.key === 'ArrowDown') { event.preventDefault(); search.current?.querySelector<HTMLAnchorElement>('.site-search-results a')?.focus(); } }} /><button aria-label="Close search" onClick={() => search.current?.close()}><X size={18} /></button></div>
        <div className="site-search-results" aria-live="polite">
          {!index && <p>{searchError ? 'Search could not load. Try opening it again.' : 'Loading search…'}</p>}
          {index && !results.length && <p>No articles match “{query}”.</p>}
          {results.map(entry => <Link href={entry.route} key={entry.route} onClick={() => search.current?.close()}><strong>{entry.title}<ArrowUpRight size={16} /></strong><span>{entry.description}</span></Link>)}
        </div><p className="site-search-hint">Tab to move through results · Escape to close</p>
      </div>
    </dialog>
  </div>;
}
