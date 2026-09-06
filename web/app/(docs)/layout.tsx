import { Footer, Layout, Navbar } from 'nextra-theme-docs';
import { getPageMap } from 'nextra/page-map';
import FooterIcons from '../../components/FooterIcons';
import { getAssetPath } from '../../lib/paths';

export default async function DocumentationLayout({ children }: { children: React.ReactNode }) {
  return <div className="docs-shell"><Layout pageMap={await getPageMap()} darkMode={false} nextThemes={{ forcedTheme: 'dark' }}
    docsRepositoryBase="https://github.com/RnLe/blaze2d/tree/main/web/content"
    sidebar={{ defaultMenuCollapseLevel: 1 }}
    navbar={<Navbar logo={<span className="site-brand"><img src={getAssetPath('/icons/blaze_bw.svg')} alt="" width={30} height={30} />Blaze2D</span>} />}
    footer={<Footer><FooterIcons /></Footer>}>
    {children}
  </Layout></div>;
}
