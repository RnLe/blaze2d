'use client';
import { useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

/** Static exports cannot send HTTP redirects; keep old bookmarks usable. */
export function WorkbenchRedirect({ href }: { href: string }) {
  const router = useRouter();
  useEffect(() => { router.replace(href); }, [href, router]);
  return <p role="status">Opening the example library in the <Link href={href}>Workbench</Link>…</p>;
}
