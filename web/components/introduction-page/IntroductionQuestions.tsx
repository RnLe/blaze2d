import { ArrowDownRight } from 'lucide-react';

const questions = [
  ['what-makes-a-crystal-photonic', 'What makes a crystal photonic?'],
  ['why-does-light-form-bands', 'Why does light form bands?'],
  ['how-do-i-read-a-band-diagram', 'How do I read a band diagram?'],
  ['how-does-blaze-solve-the-equation', 'How does Blaze solve the equation?'],
  ['why-calculate-many-different-crystals', 'Why calculate many different crystals?'],
  ['what-can-operator-data-tell-us', 'What can operator data tell us?'],
];

export default function IntroductionQuestions() {
  return <nav className="intro-questions" aria-label="Introduction questions">
    {questions.map(([id, question], index) => <a key={id} href={`#${id}`}>
      <span className="intro-question-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
      <span>{question}</span>
      <ArrowDownRight size={16} aria-hidden="true" />
    </a>)}
  </nav>;
}
