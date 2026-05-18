import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import 'katex/dist/katex.min.css'
import './global.css'
import FooterIcons from '../components/FooterIcons'

export const metadata = {
  title: 'Blaze 2D',
  description: 'A lightweight 2D Maxwell solver for photonic band structures',
}
 
const navbar = (
  <Navbar
    logo={<span style={{ color: '#ffffff', fontWeight: 600, fontSize: '1.5rem', letterSpacing: '-0.02em' }}>Blaze 2D</span>}
  />
)
const footer = <Footer><FooterIcons /></Footer>
 
export default async function RootLayout({ children }: { children: React.ReactNode }) {
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
