import '../pitch/layout.css';

export const metadata = {
  title: 'Blaze2D · Workbench',
  description: 'Design a photonic crystal, define a sweep, and solve it live in your browser.',
};

export default function WorkbenchLayout({ children }: { children: React.ReactNode }) {
  return <div className="blaze-layout">{children}</div>;
}
