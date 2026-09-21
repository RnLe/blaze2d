import Link from 'next/link';
import { ArrowUpRight, FlaskConical } from 'lucide-react';

/**
 * The strongest thing the site can offer a newcomer: a real calculation without
 * installing anything. Shaped like the sidebar's Workbench link.
 */
export default function WorkbenchCallout() {
  return (
    <Link href="/workbench" className="welcome-callout">
      <FlaskConical size={20} />
      <span className="welcome-callout-text">
        <strong>Open the Workbench</strong>
        <small>
          Use Blaze in your browser. <strong>No installation required.</strong>
        </small>
      </span>
      <ArrowUpRight size={17} />
    </Link>
  );
}
