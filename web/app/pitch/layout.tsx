import type { Metadata } from 'next';
import './pitch.css';

export const metadata: Metadata = {
  title: { absolute: 'Blaze2D: a 2D Maxwell solver' },
  description: 'A fast Rust-based 2D Maxwell solver for photonic band structures',
};

export default function PitchLayout({ children }: { children: React.ReactNode }) {
  return <div className="pitch-layout">{children}</div>;
}
