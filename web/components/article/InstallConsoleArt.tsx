/**
 * Card art for Installation: the two commands sit quietly at rest, and hovering
 * the card runs them, printing three result lines one after another.
 *
 * The result lines hold their space at rest rather than expanding into it, so
 * nothing above them shifts as they arrive. Playback is driven by
 * `.post-tile:hover` in home.css, so the card stays a server component.
 */
const COMMANDS = [
  'pip install blaze2d',
  'blaze2d config validate calculation.toml',
  'blaze2d run calculation.toml -o results.npz',
];
const RESULTS = [
  'config ok · schema blaze2d/1',
  '8 bands · TM · 64×64',
  '21 k-points · f64',
  'results.npz written',
];

export default function InstallConsoleArt() {
  return (
    <div className="console-art" aria-hidden="true">
      {COMMANDS.map(command => (
        <span className="console-art-line" key={command}>
          <i className="console-art-prompt">$</i>
          {command}
        </span>
      ))}
      {RESULTS.map(result => (
        <span className="console-art-line console-art-result" key={result}>
          <i className="console-art-tick">✓</i>
          {result}
        </span>
      ))}
    </div>
  );
}
