import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

/**
 * The four questions the site answers, as a compact index.
 *
 * It sits inside the welcome so a first-time reader can see the shape of the
 * site without scrolling, and doubles as navigation.
 */
const ENTRIES = [
  {
    route: '/installation',
    question: 'How do I use it?',
    answer: 'Install the Python package, or skip installing altogether and calculate in the browser.',
  },
  {
    route: '/blaze',
    question: 'How well does it work?',
    answer: 'Numerical methods, validation against MPB, and benchmarks with the conditions written out.',
  },
  {
    route: '/thesis',
    question: 'Why does it exist?',
    answer: 'The photonic moiré crystals that motivated the solver, and the two-scale theory behind them.',
  },
  {
    route: '/roadmap',
    question: 'Where is it going?',
    answer: 'Current scope, known limits, and the planned scientific and performance work.',
  },
];

export default function WelcomeLinks() {
  return (
    <ul className="welcome-links">
      {ENTRIES.map(entry => (
        <li key={entry.route}>
          <Link href={entry.route}>
            <strong>
              {entry.question}
              <ArrowRight size={14} aria-hidden="true" />
            </strong>
            <span>{entry.answer}</span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
