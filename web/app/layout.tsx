import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Banner, Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import 'katex/dist/katex.min.css'
import './global.css'

export const metadata = {
  title: 'Blaze 2D',
  description: 'A lightweight 2D Maxwell solver for photonic band structures',
}
 
const banner = <Banner storageKey="blaze2d-v1">Blaze 2D Documentation 📚</Banner>
const navbar = (
  <Navbar
    logo={<b>Blaze 2D</b>}
    // Additional navbar options can be added here
  />
)
const footer = <Footer>MIT {new Date().getFullYear()} © Blaze 2D.</Footer>
 
export default async function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      // Not required, but good for SEO
      lang="en"
      // Required to be set
      dir="ltr"
      // Suggested by `next-themes` package https://github.com/pacocoursey/next-themes#with-app
      suppressHydrationWarning
    >
      <Head
      // Additional head options can be added here
      >
        {/* Additional meta tags and head elements */}
      </Head>
      <body>
        <Layout
          banner={banner}
          navbar={navbar}
          pageMap={await getPageMap()}
          docsRepositoryBase="https://github.com/RnLe/blaze2d/tree/main/web/content"
          footer={footer}
          // Additional layout options
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}
