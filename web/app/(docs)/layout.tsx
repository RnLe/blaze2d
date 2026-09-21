import { DocsShell } from '@/components/site/DocsShell';
export default function DocumentationLayout({ children }: { children: React.ReactNode }) {
  return <DocsShell>{children}</DocsShell>;
}
