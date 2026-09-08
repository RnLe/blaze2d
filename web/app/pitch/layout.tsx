import './layout.css';
export const metadata = {
  title: 'Blaze 2D',
  description: 'A lightweight 2D Maxwell solver for photonic band structures',
};

export default function BlazeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="blaze-layout">
      {children}
    </div>
  );
}
