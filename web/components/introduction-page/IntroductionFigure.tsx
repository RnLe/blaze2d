import { ArrowUpRight } from 'lucide-react';
import { getAssetPath } from '@/lib/paths';

type Props = {
  src: string;
  alt: string;
  caption: string;
  width: number;
  height: number;
};

export default function IntroductionFigure({ src, alt, caption, width, height }: Props) {
  return <figure className="intro-figure">
    <a className="intro-figure-link" href={getAssetPath(src)} target="_blank" rel="noreferrer">
      <span className="intro-figure-paper"><img src={getAssetPath(src)} alt={alt} width={width} height={height} loading="lazy" /></span>
      <span className="intro-figure-action">Open full size<span className="sr-only"> in a new tab</span> <ArrowUpRight size={14} aria-hidden="true" /></span>
    </a>
    <figcaption>{caption}</figcaption>
  </figure>;
}
