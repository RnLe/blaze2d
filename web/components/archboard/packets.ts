/**
 * Data packets: pooled sprites flowing along resolved edge polylines during
 * Run playback. One shared circle texture; sprites batch in a single draw
 * call. Spawn rate per edge scales with the bytes it moves.
 */

import { Container, Graphics, RenderTexture, Sprite, type Renderer } from 'pixi.js';
import type { ResolvedEdge, SolveParams } from '../../lib/archboard/types';
import { pointAt } from '../../lib/archboard/layout';
import { PACKET_COLORS } from '../../lib/archboard/palette';

const MAX_PACKETS = 600;
const MIN_RATE = 2; // packets/s on an active edge
const MAX_RATE = 26;
/** Packet travel time along an edge (ms); shorter edges look faster anyway. */
const TRAVEL_MS = 1400;

interface Packet {
  sprite: Sprite;
  edge: ResolvedEdge;
  t: number;
  speed: number; // t units per ms
  active: boolean;
}

export class PacketSystem {
  readonly container: Container;
  private pool: Packet[] = [];
  private texture: RenderTexture;
  private spawnAccumulators = new Map<string, number>();
  private rates = new Map<string, number>();

  constructor(renderer: Renderer, params: SolveParams, edges: ResolvedEdge[]) {
    this.container = new Container();
    this.container.eventMode = 'none';

    const g = new Graphics().circle(6, 6, 5).fill(0xffffff);
    this.texture = RenderTexture.create({ width: 12, height: 12, resolution: 2 });
    renderer.render({ container: g, target: this.texture });
    g.destroy();

    // Precompute spawn rates from byte volumes (log scale).
    for (const resolved of edges) {
      const bytes = resolved.edge.bytes?.(params) ?? 0;
      const rate =
        bytes > 0
          ? Math.min(MAX_RATE, MIN_RATE + Math.max(0, Math.log10(bytes) - 2) * 4)
          : MIN_RATE;
      this.rates.set(resolved.edge.id, rate);
    }
  }

  /** Advance all live packets and spawn new ones on the active edges. */
  update(deltaMs: number, activeEdges: ResolvedEdge[] | null): void {
    for (const packet of this.pool) {
      if (!packet.active) continue;
      packet.t += packet.speed * deltaMs;
      if (packet.t >= 1) {
        packet.active = false;
        packet.sprite.visible = false;
        continue;
      }
      const [x, y] = pointAt(packet.edge, packet.t);
      packet.sprite.position.set(x, y);
    }

    if (!activeEdges || activeEdges.length === 0) return;
    for (const edge of activeEdges) {
      const rate = this.rates.get(edge.edge.id) ?? MIN_RATE;
      const acc = (this.spawnAccumulators.get(edge.edge.id) ?? 0) + (rate * deltaMs) / 1000;
      const spawns = Math.floor(acc);
      this.spawnAccumulators.set(edge.edge.id, acc - spawns);
      for (let i = 0; i < spawns; i++) this.spawn(edge);
    }
  }

  private spawn(edge: ResolvedEdge): void {
    let packet = this.pool.find((p) => !p.active);
    if (!packet) {
      if (this.pool.length >= MAX_PACKETS) return;
      const sprite = new Sprite(this.texture);
      sprite.anchor.set(0.5);
      this.container.addChild(sprite);
      packet = { sprite, edge, t: 0, speed: 0, active: false };
      this.pool.push(packet);
    }
    packet.edge = edge;
    packet.t = Math.random() * 0.06; // slight stagger
    // Constant travel time regardless of length keeps flow speeds readable.
    packet.speed = 1 / (TRAVEL_MS * (0.85 + Math.random() * 0.3));
    packet.active = true;
    packet.sprite.visible = true;
    packet.sprite.tint = PACKET_COLORS[edge.edge.kind];
    packet.sprite.alpha = 0.95;
    const scale = 0.5 + Math.random() * 0.3;
    packet.sprite.scale.set(scale);
    const [x, y] = pointAt(edge, packet.t);
    packet.sprite.position.set(x, y);
  }

  /** Kill all live packets (view switch, seek, stop). */
  clear(): void {
    for (const packet of this.pool) {
      packet.active = false;
      packet.sprite.visible = false;
    }
    this.spawnAccumulators.clear();
  }

  destroy(): void {
    this.container.destroy({ children: true });
    this.texture.destroy(true);
    this.pool = [];
  }
}
