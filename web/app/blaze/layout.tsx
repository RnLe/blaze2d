export const metadata = {
  title: 'Introduction - Blaze 2D',
  description: 'A high-performance 2D Maxwell solver for photonic band structures',
};

export default function BlazeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="blaze-intro-layout">
      {children}
    </div>
  );
}
