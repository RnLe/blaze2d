import { getAssetPath } from '@/lib/paths';

/**
 * Card art for the Manuscript: four stacked sheets that fan apart on hover over
 * 0.7s, the front one carrying the Blaze mark.
 *
 * From the `blaze-documents` package. Its geometry, masks and easing are kept;
 * four things differ:
 *
 * - The sheets are white rather than accent blue.
 * - Each sheet behind the front one carries ruled lines, so the fan reveals
 *   pages of text instead of empty outlines. They sit inside the same paper
 *   shape the masks cut against, so a sheet still hides what is behind it.
 * - The Blaze mark is an <image> rather than fifteen kilobytes of inlined path.
 * - The viewBox is cropped to the band the sheets actually occupy. The card is
 *   one grid row tall and the art is the widest thing in it, so the empty
 *   quarters above and below the fan would otherwise set the row's height.
 *
 * The masks give each sheet a hole where the sheets in front of it are, which
 * is what stops the stack looking transparent while it is closed.
 */
const PAPER = 'M122 58H182L206 82V174A8 8 0 0 1 198 182H122A8 8 0 0 1 114 174V66A8 8 0 0 1 122 58Z';
const FOLD = 'M182 58V74A8 8 0 0 0 190 82H206';
const SHEETS = [0, 1, 2, 3];

/** A sheet's silhouette, used to punch it out of the sheets behind it. */
function Cutter({ sheet }: { sheet: number }) {
  return (
    <g className={`bd-motion bd-sheet-${sheet}`}>
      <use href="#bd-paper" fill="black" stroke="black" strokeWidth="3" strokeLinejoin="round" />
    </g>
  );
}

export default function DocumentsFanArt() {
  return (
    <svg className="bd-art" viewBox="0 44 320 152" aria-hidden="true" focusable="false">
      <defs>
        <path id="bd-paper" d={PAPER} />
        <g id="bd-rules" fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round" opacity=".34">
          <path d="M130 106H188" />
          <path d="M130 122H176" />
          <path d="M130 138H186" />
          <path d="M130 154H164" />
        </g>
        {SHEETS.slice(0, 3).map(sheet => (
          <mask key={sheet} id={`bd-mask-${sheet}`} maskUnits="userSpaceOnUse" x="0" y="0" width="320" height="240" style={{ maskType: 'luminance' }}>
            <rect width="320" height="240" fill="white" />
            {SHEETS.slice(sheet + 1).map(front => <Cutter key={front} sheet={front} />)}
          </mask>
        ))}
      </defs>

      {SHEETS.slice(0, 3).map(sheet => (
        <g key={sheet} mask={`url(#bd-mask-${sheet})`}>
          <g className={`bd-motion bd-sheet-${sheet}`}>
            <g fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
              <use href="#bd-paper" />
              <path d={FOLD} />
            </g>
            <use href="#bd-rules" />
          </g>
        </g>
      ))}

      {/* The front sheet is never covered, so it needs no mask -- and carries the
          mark instead of ruled lines. */}
      <g className="bd-motion bd-sheet-3">
        <g fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <use href="#bd-paper" />
          <path d={FOLD} />
        </g>
        <image href={getAssetPath('/icons/blaze.svg')} x="137" y="90" width="46" height="74.265" />
      </g>
    </svg>
  );
}
