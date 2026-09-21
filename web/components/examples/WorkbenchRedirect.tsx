'use client';
import { useEffect } from 'react';
import { getAssetPath } from '../../lib/paths';

/** Static exports cannot send HTTP redirects; keep old bookmarks usable. */
export function WorkbenchRedirect({ href }: { href: string }) {
  const [pathname, query] = href.split('?');
  const target = getAssetPath(`${pathname.replace(/\/$/, '')}/${query ? `?${query}` : ''}`);
  useEffect(() => { window.location.replace(target); }, [target]);
  return <p role="status">Opening the example library in the <a href={target}>Workbench</a>…</p>;
}
