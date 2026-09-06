import { Head } from 'nextra/components'
import 'nextra-theme-docs/style.css'
import 'katex/dist/katex.min.css'
import './global.css'

export const metadata = {
  title: 'Blaze 2D',
  description: 'A lightweight 2D Maxwell solver for photonic band structures',
}
 

 
export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
  
  const fontCss = `
@font-face {
  font-family: 'OpenAI Sans';
  src: url('${base}/fonts/OpenAISans-Regular.woff2') format('woff2');
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: 'OpenAI Sans';
  src: url('${base}/fonts/OpenAISans-Medium.woff2') format('woff2');
  font-weight: 500;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: 'OpenAI Sans';
  src: url('${base}/fonts/OpenAISans-SemiBold.woff2') format('woff2');
  font-weight: 600;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: 'OpenAI Sans';
  src: url('${base}/fonts/OpenAISans-Bold.woff2') format('woff2');
  font-weight: 700;
  font-style: normal;
  font-display: swap;
}
`;

  return (
    <html
      lang="en"
      dir="ltr"
      suppressHydrationWarning
      className="dark"
      data-theme="dark"
    >
      <Head>
        <meta name="theme-color" content="#000000" />
        <link rel="icon" href={`${base}/favicon.ico`} sizes="any" />
        {/* Preload the self-hosted UI font weights so they are ready before
            first paint. Without this the browser only fetches a weight when it
            is first used, producing a visible font-swap flash when navigating
            between pages. */}
        <link
          rel="preload"
          href={`${base}/fonts/OpenAISans-Regular.woff2`}
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
        <link
          rel="preload"
          href={`${base}/fonts/OpenAISans-Medium.woff2`}
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
        <link
          rel="preload"
          href={`${base}/fonts/OpenAISans-SemiBold.woff2`}
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
        <link
          rel="preload"
          href={`${base}/fonts/OpenAISans-Bold.woff2`}
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
        <style dangerouslySetInnerHTML={{ __html: fontCss }} />
      </Head>
      <body
        style={{
          backgroundColor: '#000000',
          color: '#ffffff',
          fontFamily:
            "'OpenAI Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        }}
      >
        {children}
      </body>
    </html>
  )
}
