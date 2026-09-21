import type { Metadata } from 'next';
import localFont from 'next/font/local';
import 'katex/dist/katex.min.css';
import './global.css';
import { getAssetPath } from '@/lib/paths';

/**
 * next/font hashes, preloads and serves the UI font, and applies the deployment
 * base path on its own. Declaring the weights here is what stops the font-swap
 * flash that appeared when a weight was only fetched the first time it was used.
 */
const openAISans = localFont({
  src: [
    { path: './fonts/OpenAISans-Regular.woff2', weight: '400', style: 'normal' },
    { path: './fonts/OpenAISans-Medium.woff2', weight: '500', style: 'normal' },
    { path: './fonts/OpenAISans-SemiBold.woff2', weight: '600', style: 'normal' },
    { path: './fonts/OpenAISans-Bold.woff2', weight: '700', style: 'normal' },
  ],
  display: 'swap',
  variable: '--font-sans-loaded',
  fallback: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
});

export const metadata: Metadata = {
  title: { default: 'Blaze2D', template: '%s · Blaze2D' },
  description: 'A lightweight 2D Maxwell solver for photonic band structures',
  icons: { icon: getAssetPath('/favicon.ico') },
};

export const viewport = { themeColor: '#000000' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" dir="ltr" className={openAISans.variable} suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
