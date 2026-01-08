// theme.config.js
import FooterIcons from './components/FooterIcons'

export default {
  logo: <span style={{ color: '#ffffff', fontWeight: 600, fontSize: '1.5rem', letterSpacing: '-0.02em' }}>Blaze 2D</span>,
  project: { link: 'https://github.com/RnLe/blaze2d' },
  docsRepositoryBase: 'https://github.com/RnLe/blaze2d/tree/main/web/content',
  
  // Force dark mode only
  darkMode: false,
  
  // Primary color - white
  primaryHue: 0,
  primarySaturation: 0,
  
  // Footer with icons
  footer: {
    content: <FooterIcons />,
  },
  
  // Sidebar configuration
  sidebar: {
    defaultMenuCollapseLevel: 1,
    autoCollapse: false,
  },
}
