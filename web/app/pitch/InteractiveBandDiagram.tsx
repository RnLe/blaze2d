'use client';
import dynamic from 'next/dynamic';
import { examples } from '../../lib/examples/catalog.generated';
const Workbench = dynamic(() => import('../../components/workbench/Workbench'), { ssr: false });
export default function InteractiveBandDiagram() {
  return <div style={{ width: '100%', maxWidth: 1500, border: '1px solid #34403b', borderRadius: 12, overflow: 'hidden' }}>
    <Workbench initialSource={examples[0].source} title="Square rods" embedded />
  </div>;
}
