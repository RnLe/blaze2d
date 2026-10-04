import { ArrowDownRight } from 'lucide-react';

const questions = [
  ['why-does-blaze-need-a-preconditioner', 'Why does Blaze need a preconditioner?'],
  ['which-two-equations-matter', 'Which two equations matter?'],
  ['what-does-a-matrix-do-to-a-vector', 'What does a matrix do to a vector?'],
  ['when-is-a-vector-almost-an-eigenvector', 'When is a vector almost an eigenvector?'],
  ['what-does-the-preconditioner-change', 'What does the preconditioner change?'],
  ['why-must-it-be-cheap', 'Why must it be cheap?'],
  ['why-is-an-approximation-good-enough', 'Why is an approximation good enough?'],
  ['how-does-blaze-precondition-maxwells-equations', 'How does Blaze precondition Maxwell’s equations?'],
];

export default function PreconditionerQuestions() {
  return <nav className="pc-questions" aria-label="Questions on this page">
    {questions.map(([id, question], index) => <a key={id} href={`#${id}`}>
      <span className="pc-question-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
      <span>{question}</span>
      <ArrowDownRight size={16} aria-hidden="true" />
    </a>)}
  </nav>;
}
