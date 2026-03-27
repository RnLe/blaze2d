import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Banner, Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import 'katex/dist/katex.min.css'
import './global.css'

export const metadata = {
  title: 'MSL Framework',
  description: 'Documentation for the Moiré Lattice (MSL) framework',
}
 
const banner = <Banner storageKey="msl-v1">MSL v1.0 Documentation 📚</Banner>
const navbar = (
  <Navbar
    logo={<b>MSL</b>}
    // Additional navbar options can be added here
  />
)
const footer = <Footer>MIT {new Date().getFullYear()} © MSL Framework.</Footer>
 
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
          docsRepositoryBase="https://github.com/RnLe/msl/web/apps/website/app/content"
          footer={footer}
          // Additional layout options
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}
