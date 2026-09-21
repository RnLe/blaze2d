import { InstallCommand } from '@/components/site/InstallCommand';
import { PACKAGE_URL, SOURCE_URL, PROFILE_URL } from '@/components/site/SiteFooter';
import { GitHubIcon, ProfileIcon } from '@/components/site/icons';
import { getAssetPath } from '@/lib/paths';

/** The repository path, shown the way the package name is on the PyPI card. */
const REPOSITORY = SOURCE_URL.replace(/^https?:\/\//, '');

/** Installation, source code, and the author’s wider work. */
export default function InstallCards() {
  return (
    <div className="pitch-install-cards">
      <div className="pitch-install">
        <img src={getAssetPath('/icons/python.svg')} alt="" width={64} height={64} />
        <h3>Available on PyPI</h3>
        <InstallCommand size="large" />
        <a href={PACKAGE_URL} target="_blank" rel="noopener noreferrer">
          View on pypi.org →
        </a>
      </div>

      <a className="pitch-install pitch-install-link" href={SOURCE_URL} target="_blank" rel="noopener noreferrer">
        <GitHubIcon size={64} />
        <h3>Source on GitHub</h3>
        <span className="pitch-install-name">{REPOSITORY}</span>
        <span className="pitch-install-cta">View on github.com →</span>
      </a>
      <a className="pitch-install pitch-install-link" href={PROFILE_URL} target="_blank" rel="noopener noreferrer">
        <ProfileIcon size={64} />
        <h3>Author’s portfolio</h3>
        <span className="pitch-install-description">Other work by the author.</span>
        <span className="pitch-install-cta">View profile →</span>
      </a>
    </div>
  );
}
