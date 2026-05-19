// theme.config.tsx
import FooterIcons from './components/FooterIcons'
import { getAssetPath } from './lib/paths'

export default {
  logo: (
    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <img src={getAssetPath('/icons/blaze_bw.svg')} alt="Blaze" style={{ height: '1.5rem', width: 'auto' }} />
      <span style={{ color: '#ffffff', fontWeight: 600, fontSize: '1.5rem', letterSpacing: '-0.02em' }}>Blaze 2D</span>
    </span>
  ),
  project: { link: 'https://github.com/RnLe/blaze2d' },
  docsRepositoryBase: 'https://github.com/RnLe/blaze2d/tree/main/web/content',
  
  // Force dark mode only
  darkMode: false,
  
  // Primary color - white
  primaryHue: 0,
  primarySaturation: 0,
  
  // Footer with custom icons
  footer: <FooterIcons />,
  
  // Sidebar configuration
  sidebar: {
    defaultMenuCollapseLevel: 1,
    autoCollapse: false,
  },
  
  // Hide git timestamp / last updated
  gitTimestamp: null,
  
  // Hide breadcrumb/navigation
  navigation: false,
  
  // Hide edit link
  editLink: {
    component: null,
  },
  
  // Hide feedback link
  feedback: {
    content: null,
  },
}
