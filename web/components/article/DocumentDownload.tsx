import { Download } from 'lucide-react';
import { PdfIcon } from '@/components/site/icons';
import { formatDocument, getDocument } from '@/lib/documents';
import { getAssetPath } from '@/lib/paths';

/**
 * A download affordance for a PDF, shaped like the Workbench link in the
 * sidebar so the site's two "go do something" controls read as a pair.
 *
 * The length and weight come from the measured file, so a reader knows what
 * they are opening before they commit to it.
 */
export default function DocumentDownload({
  src,
  label = 'Download the PDF',
}: {
  /** Public path of the PDF, e.g. `/paper/blaze2d.pdf`. */
  src: string;
  label?: string;
}) {
  const document = getDocument(src);
  return (
    <a className="document-download" href={getAssetPath(src)} download target="_blank" rel="noreferrer">
      <PdfIcon size={20} />
      <span className="document-download-text">
        <strong>{label}</strong>
        <small>{formatDocument(document)}</small>
      </span>
      <Download size={17} />
    </a>
  );
}
