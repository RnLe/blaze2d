import { getAssetPath } from '@/lib/paths';

/**
 * Card art for API & TOML: Python and Rust sitting together, which draw apart
 * on hover to reveal the configuration document that both of them read.
 *
 * The two logos are `<image>` rather than inlined paths -- the Rust mark alone
 * is six kilobytes of path data, and neither needs recolouring from the page.
 * Everything that animates is driven by `.post-tile:hover` in home.css, so the
 * card stays a server component.
 */
export default function ApiTomlArt() {
  return (
    <svg className="api-art" viewBox="0 0 180 80" aria-hidden="true" focusable="false">
      {/* Drawn from the document outward, so the dash reveal runs that way too. */}
      <path className="api-art-line api-art-line-left" d="M77 40H53" stroke="var(--text-muted)" strokeWidth="1.8" strokeLinecap="round" fill="none" />
      <path className="api-art-line api-art-line-right" d="M103 40H127" stroke="var(--text-muted)" strokeWidth="1.8" strokeLinecap="round" fill="none" />

      <g className="api-art-document">
        <path
          d="M83 26H95L101 32V51Q101 54 98 54H83Q80 54 80 51V29Q80 26 83 26Z"
          fill="var(--accent-surface)"
          stroke="var(--accent)"
          strokeWidth="1.8"
          strokeLinejoin="round"
        />
        <path d="M95 26V32H101" fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        <g fill="var(--text-primary)" opacity=".92">
          <rect x="84" y="36" width="8" height="2.4" rx="1.2" />
          <rect x="84" y="41" width="11" height="2.4" rx="1.2" />
          <rect x="84" y="46" width="6" height="2.4" rx="1.2" />
        </g>
      </g>

      {/* At rest they sit ten units apart, centred on the document they will
          reveal; hovering carries each 34 units outwards. */}
      <image className="api-art-python" href={getAssetPath('/icons/python.svg')} x="43" y="19" width="42" height="42" />
      <image className="api-art-rust" href={getAssetPath('/icons/rust.svg')} x="95" y="19" width="42" height="42" />
    </svg>
  );
}
