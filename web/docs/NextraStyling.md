# Nextra Styling Reference

## Key CSS Variable
```css
:root {
  --nextra-content-width: 90%;  /* Default: 90rem. Use %, rem, or px */
}
```
Defined in: `node_modules/nextra/dist/client/components/head.js`

## Layout Structure
```
body
└── div (container: max-w = --nextra-content-width, mx-auto)
    ├── aside.x:w-64         ← Left sidebar (navigation)
    ├── article.x:w-full     ← Main content
    └── nav.nextra-toc       ← Right sidebar (TOC)
```

## Sidebar Selectors
```css
/* Left sidebar */
aside[class*="x:w-64"] { width: 16rem !important; }

/* Right TOC */
nav.nextra-toc { width: 16rem !important; }

/* Collapsed left sidebar */
aside[class*="x:w-20"] { width: 5rem !important; }
```

## Per-Page Options (`_meta.js`)
```js
export default {
  'page-name': {
    theme: {
      layout: 'full',      // Full-width layout
      sidebar: false,      // Hide left sidebar
      toc: false,          // Hide right TOC
      breadcrumb: false,
      pagination: false,
    }
  }
}
```

## Layout Component Props (`layout.tsx`)
```tsx
<Layout
  sidebar={{ autoCollapse: true, defaultMenuCollapseLevel: 1 }}
  toc={{ float: true, extraContent: <Component /> }}
  // ...
>
```

## Wide Content in MDX
```mdx
import { Bleed } from 'nextra/components'

<Bleed>Content breaks out of container</Bleed>
<Bleed full>Full viewport width</Bleed>
```

## CSS Class Prefix
Nextra uses `x:` prefix for Tailwind classes (e.g., `x:w-64`, `x:max-w-*`).

## Footer Configuration
**Important:** The footer is NOT configured in `theme.config.tsx`. It's defined directly in `app/layout.tsx`.

```tsx
// app/layout.tsx
import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import FooterIcons from '../components/FooterIcons'

const footer = <Footer><FooterIcons /></Footer>

export default async function RootLayout({ children }) {
  return (
    <Layout footer={footer}>
      {children}
    </Layout>
  )
}
```

The `Footer` component from `nextra-theme-docs` accepts children, so custom content (icons, links, etc.) goes inside it.

## Files
- Theme CSS: `node_modules/nextra-theme-docs/dist/style.css`
- Wrapper: `node_modules/nextra-theme-docs/dist/mdx-components/wrapper.client.js`
- Sidebar: `node_modules/nextra-theme-docs/dist/components/sidebar.js`
- TOC: `node_modules/nextra-theme-docs/dist/components/toc.js`
