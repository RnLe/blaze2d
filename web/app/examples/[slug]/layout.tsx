import '../../pitch/layout.css';

export const metadata = {
  title: 'Blaze2D · Example',
  description: 'Run a Blaze2D example live in your browser.',
};

export default function ExampleSlugLayout({ children }: { children: React.ReactNode }) {
  return <div className="blaze-layout">{children}</div>;
}
