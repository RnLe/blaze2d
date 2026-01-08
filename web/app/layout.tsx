import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import 'katex/dist/katex.min.css'
import './global.css'
import FooterIcons from '../components/FooterIcons'
import { getAssetPath } from '../lib/paths'

export const metadata = {
  title: 'Blaze 2D',
  description: 'A lightweight 2D Maxwell solver for photonic band structures',
}
 
const footer = <Footer><FooterIcons /></Footer>
 
export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const base = process.env.NEXT_PUBLIC_BASE_PATH || '';
  
  const navbar = (
    <Navbar
      logo={
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <img 
            src={getAssetPath('/icons/blaze_bw.svg')} 
            alt="Blaze 2D Logo" 
            style={{ width: '32px', height: '32px' }} 
          />
          <span style={{ color: '#ffffff', fontWeight: 600, fontSize: '1.5rem', letterSpacing: '-0.02em' }}>Blaze 2D</span>
        </div>
      }
    />
  )

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
        <style dangerouslySetInnerHTML={{ __html: fontCss }} />
      </Head>
      <body style={{ backgroundColor: '#000000', color: '#ffffff' }}>
        <Layout
          navbar={navbar}
          pageMap={await getPageMap()}
          docsRepositoryBase="https://github.com/RnLe/blaze2d/tree/main/web/content"
          footer={footer}
          darkMode={false}
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}
