import type { ReactNode } from 'react';
import { ArrowDownRight } from 'lucide-react';
import { modules, stages, type ModuleId, type ModuleStatus } from './roadmap-data';

const statusClasses: Record<ModuleStatus, string> = { Done: 'is-done', 'In Development': 'is-developing', Planned: 'is-planned' };

export function StatusLabel({ status }: { status: ModuleStatus }) {
  return <strong className={`roadmap-item-status ${statusClasses[status]}`}>{status}</strong>;
}

export function ModuleStatusTag({ module }: { module: ModuleId }) {
  return <StatusLabel status={modules[module].status} />;
}

export function Motivation({ module, children }: { module: ModuleId; children?: ReactNode }) {
  return <aside className="roadmap-motivation" aria-label={`Motivation for ${modules[module].title}`}>
    <strong className="roadmap-motivation-label">Motivation</strong>
    <p>{children ?? modules[module].motivation}</p>
  </aside>;
}

export function DevelopmentSequence() {
  return <>
    <nav aria-label="Development stages" className="roadmap-stages">
      {stages.map(stage => <section className="roadmap-stage-card" data-stage={stage.number} key={stage.number}>
        <a className="roadmap-stage-heading" href={`#${stage.anchor}`}>
          <span className="roadmap-stage-label">Stage {stage.number}<ArrowDownRight size={18} aria-hidden="true" /></span>
          <h1>{stage.title}</h1>
          <span className="roadmap-stage-description">{stage.description}</span>
        </a>
        <div className="roadmap-stage-items">
          {stage.items.map(module => {
            const item = modules[module];
            return <a className="roadmap-stage-item" href={`#${item.anchor}`} key={module}>
              <div className="roadmap-item-copy">
                <div className="roadmap-item-heading">
                  <h2>{item.title}</h2>
                  <StatusLabel status={item.status} />
                </div>
                <span className="roadmap-item-description">{item.description}</span>
              </div>
            </a>;
          })}
        </div>
      </section>)}
    </nav>
  </>;
}
