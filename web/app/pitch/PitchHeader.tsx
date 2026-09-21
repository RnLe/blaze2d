import Link from 'next/link';
import { getAssetPath } from '@/lib/paths';

/** Fixed home link in the corner of the pitch page. */
export default function PitchHeader() {
  return (
    <Link href="/" className="pitch-brand">
      <img src={getAssetPath('/icons/blaze_bw.svg')} alt="" width={24} height={24} />
      <span>Blaze 2D</span>
    </Link>
  );
}
