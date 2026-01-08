import './layout.css';
import { Inter } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  weight: ['100', '200', '300', '400', '500', '600', '700'],
  display: 'swap',
});

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
    <div className={`blaze-layout ${inter.className}`}>
      {children}
    </div>
  );
}
