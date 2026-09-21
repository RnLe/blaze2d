import { InstallCommand } from './InstallCommand';
import { GitHubIcon, PythonIcon, ProfileIcon } from './icons';

export const SOURCE_URL = 'https://github.com/RnLe/blaze2d';
export const PROFILE_URL = 'https://rnle.github.io/profile/';
export const PACKAGE_URL = 'https://pypi.org/project/blaze2d/';

/** Source, author profile, and package links under every article. */
export function SiteFooter() {
  return (
    <div className="site-footer-links">
      <a
        className="site-footer-link"
        aria-label="Blaze2D source on GitHub"
        href={SOURCE_URL}
        target="_blank"
        rel="noopener noreferrer"
      >
        <GitHubIcon />
      </a>
      <a className="site-footer-link" aria-label="Author’s profile" title="Profile" href={PROFILE_URL} target="_blank" rel="noopener noreferrer">
        <ProfileIcon />
      </a>
      <div className="site-footer-install">
        <a
          className="site-footer-link"
          aria-label="Blaze2D package on PyPI"
          href={PACKAGE_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          <PythonIcon />
        </a>
        <InstallCommand />
      </div>
    </div>
  );
}
