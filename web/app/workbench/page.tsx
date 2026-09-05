'use client';
import dynamic from 'next/dynamic';
const Workbench = dynamic(() => import('../../components/workbench/Workbench'), { ssr: false, loading: () => <p role="status">Loading Workbench…</p> });
export default function WorkbenchPage() { return <Workbench />; }
