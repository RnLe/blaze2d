import Link from 'next/link';

export const metadata = { title: 'Page not found' };

export default function NotFound() {
  return (
    <main className="not-found">
      <p>404</p>
      <h1>This page does not exist</h1>
      <p>The page may have moved, or the link may be out of date.</p>
      <Link href="/">Back to the documentation</Link>
    </main>
  );
}
