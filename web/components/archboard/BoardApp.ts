/**
 * The imperative pixi engine behind the architecture board. React never
 * touches display objects; it talks to this class through methods and reads
 * shared state from the zustand store.
 */

import { Application, Container, Graphics, Text } from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import type {
  ArchNode,
  NodeRect,
  ResolvedEdge,
  ResolvedLayout,
  RunTimeline,
  SolveParams,
} from '../../lib/archboard/types';
import { ARCH_MODEL, DEFAULT_PARAMS } from '../../lib/archboard/model';
import { resolveLayout } from '../../lib/archboard/layout';
import {
  ACCENT,
  GROUP_FILL,
  KIND_COLORS,
  NODE_FILL,
  PRECISION_COLORS,
  RUN_DIM_ALPHA,
  SURFACE,
  TEXT_SECONDARY,
} from '../../lib/archboard/palette';
import { buildTimeline, benchFromSeries4, syntheticBench, type BenchData } from '../../lib/archboard/timeline';
import { PacketSystem } from './packets';
import { useBoardStore, type BoardView } from './store';

interface NodeVisual {
  node: ArchNode;
  rect: NodeRect;
  container: Container;
  frame: Graphics;
  halo: Graphics;
  label: Text;
  isGroup: boolean;
}

interface EdgeVisual {
  resolved: ResolvedEdge;
  gfx: Graphics;
  marker: Graphics | null;
  baseWidth: number;
}

export interface BoardAppOptions {
  benchJson?: unknown;
  debugGrid?: boolean;
}

const store = useBoardStore;

export class BoardApp {
  private app!: Application;
  private viewport!: Viewport;
  private world!: Container;
  private packets!: PacketSystem;
  private layout!: ResolvedLayout;
  private timeline!: RunTimeline;
  private params: SolveParams = DEFAULT_PARAMS;

  private nodeVisuals = new Map<string, NodeVisual>();
  private edgeVisuals = new Map<string, EdgeVisual>();

  private host!: HTMLElement;
  private destroyed = false;
  private userInteracted = false;
  private view: BoardView = 'layers';
  private playing = false;
  private speed = 1;
  private playbackMs = 0;
  private lastStoreSync = 0;
  private activeStageId: string | null = null;
  private pulsePhase = 0;

  static async create(host: HTMLElement, opts: BoardAppOptions = {}): Promise<BoardApp> {
    const board = new BoardApp();
    await board.init(host, opts);
    return board;
  }

  private async init(host: HTMLElement, opts: BoardAppOptions): Promise<void> {
    this.host = host;

    const app = new Application();
    await app.init({
      preference: 'webgl',
      antialias: true,
      resolution: Math.min(globalThis.devicePixelRatio ?? 1, 2),
      autoDensity: true,
      backgroundColor: SURFACE,
      width: host.clientWidth || 800,
      height: host.clientHeight || 600,
    });
    // The host may have unmounted while init awaited.
    if (this.destroyed) {
      app.destroy(true);
      return;
    }
    this.app = app;
    host.appendChild(app.canvas);
    app.canvas.style.display = 'block';

    this.layout = resolveLayout(ARCH_MODEL);

    if (process.env.NODE_ENV !== 'production') {
      const { auditEdgeRoutes } = await import('../../lib/archboard/audit');
      const crossings = auditEdgeRoutes(this.layout, ARCH_MODEL);
      if (crossings.length > 0) {
        console.groupCollapsed(`[archboard] ${crossings.length} edge/node crossings (route: waypoints fix these)`);
        console.table(crossings);
        console.groupEnd();
      }
    }

    const bench: BenchData =
      (opts.benchJson ? benchFromSeries4(opts.benchJson) : null) ?? syntheticBench(this.params.kPoints);
    this.timeline = buildTimeline(this.params, bench);
    store.getState().setTotalMs(this.timeline.totalMs);

    this.viewport = new Viewport({
      events: app.renderer.events,
      worldWidth: this.layout.bounds.w + 200,
      worldHeight: this.layout.bounds.h + 200,
    });
    app.stage.addChild(this.viewport);
    this.viewport
      .drag()
      .pinch()
      .wheel({ smooth: 3 })
      .decelerate({ friction: 0.93 })
      .clampZoom({ minScale: 0.12, maxScale: 4 });
    // Until the user pans/zooms, window resizes keep the whole board fitted.
    this.viewport.on('drag-start', () => (this.userInteracted = true));
    this.viewport.on('wheel-start', () => (this.userInteracted = true));
    this.viewport.on('pinch-start', () => (this.userInteracted = true));

    this.world = new Container();
    this.viewport.addChild(this.world);

    if (opts.debugGrid) this.buildDebugGrid();
    this.buildEdges();
    this.buildNodes();

    this.packets = new PacketSystem(app.renderer, this.params, this.layout.edges);
    this.world.addChild(this.packets.container);

    // Deselect on empty-space click (viewport 'clicked' fires when no child handled it).
    this.viewport.on('clicked', () => this.selectNode(null));

    // Label crispness: re-rasterize visible labels after zoom settles.
    let zoomTimer: ReturnType<typeof setTimeout> | null = null;
    const onZoom = () => {
      if (zoomTimer) clearTimeout(zoomTimer);
      zoomTimer = setTimeout(() => this.refreshTextResolution(), 180);
      this.applyLabelLod();
    };
    this.viewport.on('zoomed', onZoom);

    app.ticker.add(() => this.tick(app.ticker.deltaMS));

    document.addEventListener('visibilitychange', this.onVisibility);

    this.fitAll(false);
    store.getState().setReady(true);
  }

  /* --------------------------------- scene -------------------------------- */

  private buildDebugGrid(): void {
    const g = new Graphics();
    const { bounds } = this.layout;
    for (let x = 0; x <= bounds.w + 100; x += 16 * 4) {
      g.moveTo(x, 0).lineTo(x, bounds.h + 100);
    }
    for (let y = 0; y <= bounds.h + 100; y += 16 * 4) {
      g.moveTo(0, y).lineTo(bounds.w + 100, y);
    }
    g.stroke({ width: 1, color: 0x222222, alpha: 0.5 });
    this.world.addChild(g);
  }

  private buildEdges(): void {
    for (const resolved of this.layout.edges) {
      const { edge } = resolved;
      const bytes = edge.bytes?.(this.params) ?? 0;
      const baseWidth =
        edge.kind === 'control'
          ? 1.25
          : Math.min(6, 1.5 + Math.max(0, Math.log10(Math.max(bytes, 1)) - 2) * 0.55);
      const color = edge.kind === 'control' ? KIND_COLORS.control : 0x4a7fb5;

      const gfx = new Graphics();
      this.drawEdgePath(gfx, resolved, baseWidth, color, edge.kind === 'control');
      gfx.alpha = edge.kind === 'control' ? 0.55 : 0.8;
      this.world.addChild(gfx);

      let marker: Graphics | null = null;
      if (edge.precisionChange) {
        marker = new Graphics();
        const mid = this.edgeMidpoint(resolved);
        marker.rect(-5, -5, 10, 10).fill(0xffffff);
        marker.rotation = Math.PI / 4;
        marker.position.set(mid[0], mid[1]);
        marker.alpha = 0.9;
        this.world.addChild(marker);
      }

      this.edgeVisuals.set(edge.id, { resolved, gfx, marker, baseWidth });
    }
  }

  private edgeMidpoint(resolved: ResolvedEdge): [number, number] {
    const half = resolved.totalLength / 2;
    const { points, arc } = resolved;
    for (let i = 1; i < points.length; i++) {
      if (half <= arc[i]) {
        const t = (half - arc[i - 1]) / (arc[i] - arc[i - 1] || 1);
        return [
          points[i - 1][0] + (points[i][0] - points[i - 1][0]) * t,
          points[i - 1][1] + (points[i][1] - points[i - 1][1]) * t,
        ];
      }
    }
    return points[points.length - 1];
  }

  private drawEdgePath(
    gfx: Graphics,
    resolved: ResolvedEdge,
    width: number,
    color: number,
    dashed: boolean
  ): void {
    gfx.clear();
    const { points } = resolved;
    if (dashed) {
      const dash = 7;
      const gap = 5;
      for (let i = 1; i < points.length; i++) {
        const [x0, y0] = points[i - 1];
        const [x1, y1] = points[i];
        const len = Math.hypot(x1 - x0, y1 - y0);
        const ux = (x1 - x0) / len;
        const uy = (y1 - y0) / len;
        let dist = 0;
        while (dist < len) {
          const end = Math.min(dist + dash, len);
          gfx.moveTo(x0 + ux * dist, y0 + uy * dist).lineTo(x0 + ux * end, y0 + uy * end);
          dist = end + gap;
        }
      }
    } else {
      gfx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) gfx.lineTo(points[i][0], points[i][1]);
    }
    gfx.stroke({ width, color, alpha: 1 });

    // Arrowhead on the final segment.
    const [ax, ay] = points[points.length - 1];
    const [px, py] = points[points.length - 2];
    const angle = Math.atan2(ay - py, ax - px);
    const size = Math.max(6, width * 2.2);
    gfx
      .moveTo(ax, ay)
      .lineTo(ax - size * Math.cos(angle - 0.42), ay - size * Math.sin(angle - 0.42))
      .lineTo(ax - size * Math.cos(angle + 0.42), ay - size * Math.sin(angle + 0.42))
      .closePath()
      .fill(color);
  }

  private buildNodes(): void {
    // Draw order: groups by depth first (frames behind), then leaves.
    const sorted = [...ARCH_MODEL.nodes].sort(
      (a, b) => (this.layout.depth.get(a.id) ?? 0) - (this.layout.depth.get(b.id) ?? 0)
    );

    for (const node of sorted) {
      const rect = this.layout.rects.get(node.id)!;
      const isGroup = node.type === 'group';
      const container = new Container();
      container.position.set(rect.x, rect.y);

      const frame = new Graphics();
      const halo = new Graphics();
      this.drawNodeFrame(frame, node, rect, false);
      halo.roundRect(-4, -4, rect.w + 8, rect.h + 8, 12).stroke({ width: 4, color: this.kindColor(node) });
      halo.alpha = 0;

      const label = new Text({
        text: node.label,
        style: {
          fontFamily: 'Inter, system-ui, sans-serif',
          fontSize: isGroup ? 16 : 13.5,
          fontWeight: isGroup ? '700' : '500',
          fill: isGroup ? TEXT_SECONDARY : 0xf2f2f2,
          wordWrap: !isGroup,
          wordWrapWidth: rect.w - 12,
          align: isGroup ? 'left' : 'center',
          lineHeight: isGroup ? 19 : 16,
        },
        resolution: 2,
      });
      if (isGroup) {
        label.position.set(10, 7);
      } else {
        label.anchor.set(0.5);
        label.position.set(rect.w / 2, rect.h / 2);
      }

      container.addChild(halo, frame, label);

      if (node.badge) {
        const badgeText = new Text({
          text: node.badge,
          style: {
            fontFamily: 'Inter, system-ui, sans-serif',
            fontSize: 9,
            fontWeight: '600',
            fill: 0xffffff,
          },
        });
        const pad = 4;
        const bw = badgeText.width + pad * 2;
        const bh = badgeText.height + 3;
        const badgeBg = new Graphics()
          .roundRect(0, 0, bw, bh, 4)
          .fill({ color: this.kindColor(node), alpha: 0.35 })
          .stroke({ width: 1, color: this.kindColor(node), alpha: 0.8 });
        badgeBg.position.set(rect.w - bw - 4, -bh / 2);
        badgeText.position.set(rect.w - bw - 4 + pad, -bh / 2 + 1.5);
        container.addChild(badgeBg, badgeText);
      }

      // Interaction (leaves and groups both hoverable/clickable; group hit area
      // is its title strip so children stay reachable).
      container.eventMode = 'static';
      container.cursor = 'pointer';
      if (isGroup) {
        frame.eventMode = 'none';
        label.eventMode = 'static';
        label.cursor = 'pointer';
        container.eventMode = 'passive';
        label.on('pointerover', (e) => this.onHover(node, e.global.x, e.global.y));
        label.on('pointerout', () => store.getState().setHover(null));
        label.on('pointertap', (e) => {
          e.stopPropagation();
          this.selectNode(node.id);
        });
      } else {
        container.on('pointerover', (e) => this.onHover(node, e.global.x, e.global.y));
        container.on('pointerout', () => store.getState().setHover(null));
        container.on('pointertap', (e) => {
          e.stopPropagation();
          this.selectNode(node.id);
        });
      }

      this.world.addChild(container);
      this.nodeVisuals.set(node.id, { node, rect, container, frame, halo, label, isGroup });
    }
  }

  private kindColor(node: ArchNode): number {
    return KIND_COLORS[node.kind];
  }

  private drawNodeFrame(gfx: Graphics, node: ArchNode, rect: NodeRect, highlighted: boolean): void {
    gfx.clear();
    const isGroup = node.type === 'group';
    const radius = node.type === 'buffer' ? Math.min(16, rect.h / 2) : isGroup ? 10 : 8;
    const color = highlighted ? ACCENT : this.kindColor(node);
    const borderWidth = highlighted ? 2.5 : isGroup ? 1.25 : 1.5;

    if (node.stub) {
      gfx.roundRect(0, 0, rect.w, rect.h, radius).fill({ color: NODE_FILL, alpha: 0.5 });
      // dashed border
      const dash = 6;
      const gap = 5;
      const perim = [
        [0, 0, rect.w, 0],
        [rect.w, 0, rect.w, rect.h],
        [rect.w, rect.h, 0, rect.h],
        [0, rect.h, 0, 0],
      ] as const;
      for (const [x0, y0, x1, y1] of perim) {
        const len = Math.hypot(x1 - x0, y1 - y0);
        const ux = (x1 - x0) / len;
        const uy = (y1 - y0) / len;
        let d = 0;
        while (d < len) {
          const end = Math.min(d + dash, len);
          gfx.moveTo(x0 + ux * d, y0 + uy * d).lineTo(x0 + ux * end, y0 + uy * end);
          d = end + gap;
        }
      }
      gfx.stroke({ width: borderWidth, color, alpha: 0.7 });
      return;
    }

    gfx
      .roundRect(0, 0, rect.w, rect.h, radius)
      .fill({ color: isGroup ? GROUP_FILL : NODE_FILL, alpha: isGroup ? 0.72 : 1 })
      .stroke({ width: borderWidth, color, alpha: isGroup ? 0.55 : 0.95 });

    // Ports get a second inner border: the "socket" look.
    if (node.type === 'port') {
      gfx
        .roundRect(3, 3, rect.w - 6, rect.h - 6, Math.max(4, radius - 3))
        .stroke({ width: 1, color, alpha: 0.5 });
    }
    // Group title separator line.
    if (isGroup) {
      gfx
        .moveTo(1, 30)
        .lineTo(rect.w - 1, 30)
        .stroke({ width: 1, color, alpha: 0.25 });
    }
  }

  /* ------------------------------ interaction ----------------------------- */

  private onHover(node: ArchNode, globalX: number, globalY: number): void {
    const canvasRect = this.app.canvas.getBoundingClientRect();
    const hostRect = this.host.getBoundingClientRect();
    store.getState().setHover({
      id: node.id,
      x: globalX + (canvasRect.left - hostRect.left),
      y: globalY + (canvasRect.top - hostRect.top),
    });
  }

  selectNode(id: string | null, fly = true): void {
    const previous = store.getState().selected;
    if (previous) {
      const prevVisual = this.nodeVisuals.get(previous);
      if (prevVisual) this.drawNodeFrame(prevVisual.frame, prevVisual.node, { ...prevVisual.rect, x: 0, y: 0 } as NodeRect, false);
    }
    store.getState().setSelected(id);
    if (!id) return;
    const visual = this.nodeVisuals.get(id);
    if (!visual) return;
    this.drawNodeFrame(visual.frame, visual.node, { ...visual.rect, x: 0, y: 0 } as NodeRect, true);
    if (fly) this.flyTo(visual.rect);
  }

  private flyTo(rect: NodeRect): void {
    const screenW = this.app.renderer.width / this.app.renderer.resolution;
    const screenH = this.app.renderer.height / this.app.renderer.resolution;
    const scale = Math.min(2.2, Math.min(screenW / (rect.w * 3.2), screenH / (rect.h * 6)));
    this.viewport.animate({
      time: 650,
      position: { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 },
      scale,
      ease: 'easeInOutSine',
    });
  }

  fitAll(animate = true): void {
    const { bounds } = this.layout;
    const screenW = this.app.renderer.width / this.app.renderer.resolution;
    const screenH = this.app.renderer.height / this.app.renderer.resolution;
    const scale = Math.min(screenW / (bounds.w + 120), screenH / (bounds.h + 120));
    const position = { x: bounds.x + bounds.w / 2, y: bounds.y + bounds.h / 2 };
    if (animate) {
      this.viewport.animate({ time: 500, position, scale, ease: 'easeInOutSine' });
    } else {
      this.viewport.setZoom(scale, true);
      this.viewport.moveCenter(position.x, position.y);
      this.refreshTextResolution();
      this.applyLabelLod();
    }
  }

  zoomBy(factor: number): void {
    this.viewport.zoomPercent(factor - 1, true);
  }

  panBy(dx: number, dy: number): void {
    this.viewport.moveCenter(this.viewport.center.x + dx / this.viewport.scale.x, this.viewport.center.y + dy / this.viewport.scale.y);
  }

  resize(): void {
    if (this.destroyed || !this.app) return;
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (w > 0 && h > 0) {
      this.app.renderer.resize(w, h);
      this.viewport.resize(w, h);
      if (!this.userInteracted) this.fitAll(false);
    }
  }

  private refreshTextResolution(): void {
    const zoom = this.viewport.scale.x;
    const target = Math.min(4, Math.max(2, Math.ceil((globalThis.devicePixelRatio ?? 1) * zoom)));
    for (const visual of this.nodeVisuals.values()) {
      if (visual.label.visible) visual.label.resolution = target;
    }
  }

  private applyLabelLod(): void {
    const zoom = this.viewport.scale.x;
    for (const visual of this.nodeVisuals.values()) {
      if (visual.isGroup) {
        visual.label.visible = true;
      } else {
        visual.label.visible = zoom > 0.25;
      }
    }
  }

  /* --------------------------------- views -------------------------------- */

  setView(view: BoardView): void {
    this.view = view;
    store.getState().setView(view);
    if (view !== 'run') {
      this.pause();
      this.packets.clear();
      this.playbackMs = 0;
      store.getState().setPlayback(0, null, null);
      this.setAllActive(null);
    }
    this.applyViewTint();
  }

  private applyViewTint(): void {
    for (const visual of this.nodeVisuals.values()) {
      const { node, frame, rect } = visual;
      const local = { ...rect, x: 0, y: 0 } as NodeRect;
      if (this.view === 'precision') {
        frame.clear();
        const isGroup = node.type === 'group';
        const radius = node.type === 'buffer' ? Math.min(16, rect.h / 2) : isGroup ? 10 : 8;
        if (node.precision) {
          const color = PRECISION_COLORS[node.precision];
          frame
            .roundRect(0, 0, local.w, local.h, radius)
            .fill({ color: NODE_FILL, alpha: 1 })
            .stroke({ width: 2, color, alpha: 1 });
          if (node.precision === 'mixed') {
            // Split fill: teal left half, gold right half, subtle.
            frame.roundRect(0, 0, local.w / 2, local.h, radius).fill({ color: PRECISION_COLORS.f32, alpha: 0.14 });
            frame.roundRect(local.w / 2, 0, local.w / 2, local.h, radius).fill({ color: PRECISION_COLORS.f64, alpha: 0.14 });
          } else {
            frame.roundRect(0, 0, local.w, local.h, radius).fill({ color, alpha: 0.1 });
          }
          visual.container.alpha = 1;
        } else {
          this.drawNodeFrame(frame, node, local, false);
          visual.container.alpha = isGroup ? 0.75 : 0.3;
        }
      } else {
        this.drawNodeFrame(frame, node, local, store.getState().selected === node.id);
        visual.container.alpha = 1;
      }
    }
    for (const edgeVisual of this.edgeVisuals.values()) {
      if (edgeVisual.marker) {
        edgeVisual.marker.scale.set(this.view === 'precision' ? 1.6 : 1);
        edgeVisual.marker.alpha = this.view === 'precision' ? 1 : 0.9;
      }
      if (this.view === 'precision') {
        edgeVisual.gfx.alpha = edgeVisual.resolved.edge.precisionChange ? 1 : 0.18;
      } else {
        edgeVisual.gfx.alpha = edgeVisual.resolved.edge.kind === 'control' ? 0.55 : 0.8;
      }
    }
  }

  /* ------------------------------- playback ------------------------------- */

  play(): void {
    if (this.view !== 'run') this.setView('run');
    if (this.playbackMs >= this.timeline.totalMs) this.playbackMs = 0;
    this.playing = true;
    store.getState().setPlaying(true);
  }

  pause(): void {
    this.playing = false;
    store.getState().setPlaying(false);
  }

  setSpeed(speed: number): void {
    this.speed = speed;
    store.getState().setSpeed(speed);
  }

  seek(ms: number): void {
    this.playbackMs = Math.max(0, Math.min(ms, this.timeline.totalMs));
    this.packets.clear();
    this.syncStage(true);
  }

  getTimeline(): RunTimeline {
    return this.timeline;
  }

  private tick(deltaMs: number): void {
    if (this.destroyed) return;
    this.pulsePhase += deltaMs / 300;

    let activeEdges: ResolvedEdge[] | null = null;

    if (this.view === 'run') {
      if (this.playing) {
        this.playbackMs += deltaMs * this.speed;
        if (this.playbackMs >= this.timeline.totalMs) {
          this.playbackMs = this.timeline.totalMs;
          this.pause();
        }
      }
      const flat = this.timeline.flat;
      const current = flat.find((f) => this.playbackMs >= f.start && this.playbackMs < f.end) ?? flat[flat.length - 1];
      if (current.stage.id !== this.activeStageId) {
        this.activeStageId = current.stage.id;
        this.syncStage(false);
      }
      if (this.playing) {
        activeEdges = current.stage.edges
          .map((id) => this.edgeVisuals.get(id)?.resolved)
          .filter((e): e is ResolvedEdge => Boolean(e));
      }
      // Pulse active nodes.
      const pulse = 0.45 + 0.35 * Math.sin(this.pulsePhase * 2);
      for (const nodeId of current.stage.nodes) {
        const visual = this.nodeVisuals.get(nodeId);
        if (visual) visual.halo.alpha = pulse;
      }
      // Throttled store sync for the scrubber.
      if (performance.now() - this.lastStoreSync > 120) {
        this.lastStoreSync = performance.now();
        store.getState().setPlayback(this.playbackMs, current.stage.label, current.stage.caption ?? null);
      }
    }

    this.packets.update(deltaMs, activeEdges);
  }

  /** Re-applies active/dim states after a stage change or seek. */
  private syncStage(force: boolean): void {
    const flat = this.timeline.flat;
    const current = flat.find((f) => this.playbackMs >= f.start && this.playbackMs < f.end) ?? flat[flat.length - 1];
    this.activeStageId = current.stage.id;
    this.setAllActive(new Set([...current.stage.nodes]));
    const activeEdgeSet = new Set(current.stage.edges);
    for (const [id, edgeVisual] of this.edgeVisuals) {
      edgeVisual.gfx.alpha = activeEdgeSet.has(id) ? 1 : RUN_DIM_ALPHA;
      if (edgeVisual.marker) edgeVisual.marker.alpha = activeEdgeSet.has(id) ? 1 : RUN_DIM_ALPHA;
    }
    if (force) {
      store.getState().setPlayback(this.playbackMs, current.stage.label, current.stage.caption ?? null);
    }
  }

  /** null = everything active (layers view); a set = run-view dimming. */
  private setAllActive(activeNodes: Set<string> | null): void {
    for (const [id, visual] of this.nodeVisuals) {
      if (activeNodes === null) {
        visual.container.alpha = 1;
        visual.halo.alpha = 0;
      } else {
        // Groups stay faintly visible for orientation.
        const active = activeNodes.has(id);
        visual.container.alpha = active ? 1 : visual.isGroup ? 0.5 : RUN_DIM_ALPHA;
        if (!active) visual.halo.alpha = 0;
      }
    }
    if (activeNodes === null) {
      for (const edgeVisual of this.edgeVisuals.values()) {
        edgeVisual.gfx.alpha = edgeVisual.resolved.edge.kind === 'control' ? 0.55 : 0.8;
        if (edgeVisual.marker) edgeVisual.marker.alpha = 0.9;
      }
    }
  }

  /* -------------------------------- lifecycle ------------------------------ */

  private onVisibility = (): void => {
    if (!this.app) return;
    if (document.hidden) this.app.ticker.stop();
    else this.app.ticker.start();
  };

  setRunning(running: boolean): void {
    if (!this.app) return;
    if (running) this.app.ticker.start();
    else this.app.ticker.stop();
  }

  destroy(): void {
    this.destroyed = true;
    document.removeEventListener('visibilitychange', this.onVisibility);
    if (this.app) {
      this.packets?.destroy();
      this.app.destroy(true, { children: true, texture: true });
    }
    this.nodeVisuals.clear();
    this.edgeVisuals.clear();
  }
}
