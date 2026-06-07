import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import type { GraphEdgeDto, GraphNodeDto, GraphNodeType } from '../api';

interface GraphCanvasProps {
  nodes: GraphNodeDto[];
  edges: GraphEdgeDto[];
  selectedNodeId?: string | null;
  onSelectNode: (nodeId: string) => void;
}

interface PositionedNode {
  node: GraphNodeDto;
  x: number;
  y: number;
}

interface ViewBox {
  minX: number;
  minY: number;
  width: number;
  height: number;
}

interface GraphLayout {
  positioned: PositionedNode[];
  viewBox: ViewBox;
}

const NODE_WIDTH = 132;
const NODE_HEIGHT = 44;
// Desired edge length in layout units. Nodes settle roughly this far apart, so it
// must comfortably exceed the node footprint to keep boxes from overlapping.
const IDEAL_DISTANCE = 215;
// Smallest viewBox the canvas zooms to, so a 1-2 node graph is not absurdly magnified.
const MIN_FRAME = 760;
// Keep the canvas landscape-ish regardless of the (roughly circular) layout cloud.
const MIN_ASPECT = 1.45;

// Pan / zoom interaction bounds. The viewport transform is applied on top of the
// fitted viewBox, so scale 1 always frames the whole graph.
const MIN_SCALE = 0.4;
const MAX_SCALE = 8;
const ZOOM_STEP = 1.3;

interface ViewTransform {
  scale: number;
  tx: number;
  ty: number;
}

const IDENTITY_VIEW: ViewTransform = { scale: 1, tx: 0, ty: 0 };

// Zoom toward an anchor point (in viewBox units) so the content under the cursor
// stays put: a child point p maps to scale*p + t, and we hold (anchor) fixed.
function zoomTowards(prev: ViewTransform, factor: number, anchorX: number, anchorY: number): ViewTransform {
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, prev.scale * factor));
  const ratio = scale / prev.scale;
  return {
    scale,
    tx: anchorX - ratio * (anchorX - prev.tx),
    ty: anchorY - ratio * (anchorY - prev.ty),
  };
}

const NODE_TYPE_ORDER: Record<GraphNodeType, number> = {
  document: 0,
  topic: 1,
  tag: 2,
  source: 3,
};

function sortedNodes(nodes: GraphNodeDto[]): GraphNodeDto[] {
  return [...nodes].sort((a, b) => {
    const typeDelta = NODE_TYPE_ORDER[a.type] - NODE_TYPE_ORDER[b.type];
    if (typeDelta !== 0) return typeDelta;
    const aKey = `${a.label}:${a.id}`;
    const bKey = `${b.label}:${b.id}`;
    if (aKey < bKey) return -1;
    if (aKey > bKey) return 1;
    return 0;
  });
}

interface Point {
  x: number;
  y: number;
}

// Fit a set of node centers into a padded, landscape viewBox. Because the canvas
// SVG scales to a fixed on-screen width, a larger cloud (more nodes) renders each
// node smaller — which is exactly what we want for dense graphs.
function fitViewBox(points: Point[]): ViewBox {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const { x, y } of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const padding = NODE_WIDTH;
  minX -= NODE_WIDTH / 2 + padding;
  maxX += NODE_WIDTH / 2 + padding;
  minY -= NODE_HEIGHT / 2 + padding;
  maxY += NODE_HEIGHT / 2 + padding;

  let width = Math.max(maxX - minX, MIN_FRAME);
  let height = Math.max(maxY - minY, MIN_FRAME / MIN_ASPECT);

  // Grow the shorter axis to enforce a minimum landscape aspect ratio, keeping the
  // existing content centered.
  if (width / height < MIN_ASPECT) {
    const target = height * MIN_ASPECT;
    minX -= (target - width) / 2;
    width = target;
  }

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return { minX: cx - width / 2, minY: cy - height / 2, width, height };
}

// Deterministic Fruchterman-Reingold force-directed layout. Connected nodes are
// pulled together (springs along edges) while every pair repels, so the graph
// unfolds into a readable cloud instead of a single overcrowded ring. Initial
// positions come from a golden-angle spiral (no randomness) so the result is
// stable across renders and testable.
function forceLayout(nodes: GraphNodeDto[], edges: GraphEdgeDto[]): GraphLayout {
  const ordered = sortedNodes(nodes);
  const n = ordered.length;
  if (n === 0) {
    return { positioned: [], viewBox: { minX: 0, minY: 0, width: MIN_FRAME, height: MIN_FRAME / MIN_ASPECT } };
  }
  if (n === 1) {
    return {
      positioned: [{ node: ordered[0], x: 0, y: 0 }],
      viewBox: fitViewBox([{ x: 0, y: 0 }]),
    };
  }

  const indexById = new Map<string, number>();
  ordered.forEach((node, index) => indexById.set(node.id, index));

  // Reference scale so equilibrium spacing between nodes is ~IDEAL_DISTANCE. A center
  // gravity (below) balances all-pairs repulsion to settle the cloud at radius
  // ~k*sqrt(n/gravity) — a filled disk, instead of a tight ring (no gravity) or a
  // hollow rectangle pinned to the walls (hard clamping).
  const k = IDEAL_DISTANCE;
  const frame = Math.max(MIN_FRAME, k * Math.sqrt(n));
  const half = frame / 2;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const pos: Point[] = ordered.map((_, i) => {
    const radius = half * 0.7 * Math.sqrt((i + 0.5) / n);
    const angle = i * goldenAngle;
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  });

  const links: Array<[number, number]> = [];
  for (const edge of edges) {
    const s = indexById.get(edge.source);
    const t = indexById.get(edge.target);
    if (s !== undefined && t !== undefined && s !== t) links.push([s, t]);
  }

  const repulsion = k * k;
  // Center gravity strength. Equilibrium radius ~= k*sqrt(n/gravity); ~2 fills the
  // disk to roughly 0.7*frame without flinging nodes to the boundary.
  const gravity = 2;
  // Below this distance, a strong short-range push keeps overlapping boxes apart even
  // around high-degree hubs (e.g. a topic node tied to every one of its documents).
  const minSeparation = NODE_WIDTH * 1.2;
  const safetyBound = frame * 1.6;
  const iterations = n <= 60 ? 320 : 460;
  const initialTemp = frame * 0.1;

  for (let step = 0; step < iterations; step += 1) {
    const disp: Point[] = pos.map(() => ({ x: 0, y: 0 }));

    // Repulsive force between every pair of nodes: fr(d) = k^2 / d.
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        let dx = pos[i].x - pos[j].x;
        let dy = pos[i].y - pos[j].y;
        let distSq = dx * dx + dy * dy;
        if (distSq < 0.01) {
          // Deterministic nudge for coincident points (golden-angle init makes this rare).
          dx = ((i % 7) - 3) / 3 || 1;
          dy = ((j % 5) - 2) / 2 || 1;
          distSq = dx * dx + dy * dy;
        }
        const dist = Math.sqrt(distSq);
        let force = repulsion / dist;
        if (dist < minSeparation) {
          // Extra short-range push (grows as boxes get closer) to resolve overlap.
          force += (repulsion * (minSeparation - dist)) / (dist * minSeparation);
        }
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        disp[i].x += fx;
        disp[i].y += fy;
        disp[j].x -= fx;
        disp[j].y -= fy;
      }
    }

    // Attractive spring force along each edge: fa(d) = d^2 / k.
    for (const [s, t] of links) {
      const dx = pos[s].x - pos[t].x;
      const dy = pos[s].y - pos[t].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const force = (dist * dist) / k;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      disp[s].x -= fx;
      disp[s].y -= fy;
      disp[t].x += fx;
      disp[t].y += fy;
    }

    // Pull every node toward the origin (force ~ gravity * radius) so repulsion cannot
    // push the cloud onto the walls, then apply displacement capped by a cooling
    // temperature. A generous safety bound only catches numerical runaways.
    const temperature = initialTemp * (1 - step / iterations);
    for (let i = 0; i < n; i += 1) {
      disp[i].x -= pos[i].x * gravity;
      disp[i].y -= pos[i].y * gravity;
      const len = Math.sqrt(disp[i].x * disp[i].x + disp[i].y * disp[i].y) || 1;
      const capped = Math.min(len, temperature);
      pos[i].x = Math.max(-safetyBound, Math.min(safetyBound, pos[i].x + (disp[i].x / len) * capped));
      pos[i].y = Math.max(-safetyBound, Math.min(safetyBound, pos[i].y + (disp[i].y / len) * capped));
    }
  }

  const positioned = ordered.map((node, i) => ({ node, x: pos[i].x, y: pos[i].y }));
  return { positioned, viewBox: fitViewBox(pos) };
}

function immediateNeighborIds(edges: GraphEdgeDto[], selectedNodeId?: string | null): Set<string> {
  const related = new Set<string>();
  if (!selectedNodeId) return related;

  related.add(selectedNodeId);
  for (const edge of edges) {
    if (edge.source === selectedNodeId) related.add(edge.target);
    if (edge.target === selectedNodeId) related.add(edge.source);
  }
  return related;
}

function truncateLabel(label: string): string {
  return label.length > 16 ? `${label.slice(0, 15)}…` : label;
}

function handleControlKey(event: KeyboardEvent<SVGGElement>, onSelect: () => void): void {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  event.preventDefault();
  onSelect();
}

export default function GraphCanvas({ nodes, edges, selectedNodeId, onSelectNode }: GraphCanvasProps) {
  const { positioned: positionedNodes, viewBox } = useMemo(() => forceLayout(nodes, edges), [nodes, edges]);
  const positionsById = useMemo(() => {
    return new Map(positionedNodes.map((entry) => [entry.node.id, entry]));
  }, [positionedNodes]);
  const relatedNodeIds = useMemo(() => immediateNeighborIds(edges, selectedNodeId), [edges, selectedNodeId]);
  // The type sub-label only stays legible when nodes are large (sparse graphs); the
  // box fill colour already encodes type, so drop it on dense graphs to cut clutter.
  const showTypeLabel = positionedNodes.length <= 40;

  const svgRef = useRef<SVGSVGElement>(null);
  const [view, setView] = useState<ViewTransform>(IDENTITY_VIEW);
  const [isPanning, setIsPanning] = useState(false);
  // Mirror the live viewBox into a ref so the native wheel listener and pan handlers
  // always read current values without needing to rebind.
  const viewBoxRef = useRef(viewBox);
  viewBoxRef.current = viewBox;

  // A fresh layout (e.g. switching topic/depth) reframes the whole graph.
  useEffect(() => {
    setView(IDENTITY_VIEW);
  }, [viewBox.minX, viewBox.minY, viewBox.width, viewBox.height]);

  // Translate a client (pixel) point into viewBox coordinates. The canvas uses
  // height:auto so the rendered aspect ratio matches the viewBox — no letterboxing.
  const toViewBoxPoint = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    const vb = viewBoxRef.current;
    if (!svg) return { x: vb.minX + vb.width / 2, y: vb.minY + vb.height / 2 };
    const rect = svg.getBoundingClientRect();
    return {
      x: vb.minX + ((clientX - rect.left) / rect.width) * vb.width,
      y: vb.minY + ((clientY - rect.top) / rect.height) * vb.height,
    };
  }, []);

  // Wheel-to-zoom, anchored on the cursor. Registered natively so we can preventDefault
  // (React marks wheel listeners passive, which would let the page scroll instead).
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return undefined;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const anchor = toViewBoxPoint(event.clientX, event.clientY);
      const factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
      setView((prev) => zoomTowards(prev, factor, anchor.x, anchor.y));
    };
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, [toViewBoxPoint]);

  // Drag-to-pan from empty canvas (pointerdowns on a node fall through to its click).
  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (event.button !== 0) return;
      if ((event.target as Element).closest('.graph-node')) return;
      let last = { x: event.clientX, y: event.clientY };
      setIsPanning(true);
      const onMove = (move: PointerEvent) => {
        const svg = svgRef.current;
        if (!svg) return;
        const rect = svg.getBoundingClientRect();
        const vb = viewBoxRef.current;
        const dx = ((move.clientX - last.x) / rect.width) * vb.width;
        const dy = ((move.clientY - last.y) / rect.height) * vb.height;
        last = { x: move.clientX, y: move.clientY };
        setView((prev) => ({ ...prev, tx: prev.tx + dx, ty: prev.ty + dy }));
      };
      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        setIsPanning(false);
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    },
    [],
  );

  const zoomFromButton = useCallback((factor: number) => {
    const vb = viewBoxRef.current;
    setView((prev) => zoomTowards(prev, factor, vb.minX + vb.width / 2, vb.minY + vb.height / 2));
  }, []);
  const resetView = useCallback(() => setView(IDENTITY_VIEW), []);

  return (
    <div className="graph-canvas-frame">
      <div className="graph-canvas-controls">
        <button aria-label="Zoom in" className="pixel-button" onClick={() => zoomFromButton(ZOOM_STEP)} type="button">
          +
        </button>
        <button
          aria-label="Zoom out"
          className="pixel-button"
          onClick={() => zoomFromButton(1 / ZOOM_STEP)}
          type="button"
        >
          −
        </button>
        <button aria-label="Reset view" className="pixel-button" onClick={resetView} type="button">
          Fit
        </button>
      </div>
      <svg
        aria-label="Knowledge graph"
        className={`graph-canvas ${isPanning ? 'is-panning' : ''}`}
        onPointerDown={handlePointerDown}
        ref={svgRef}
        role="group"
        viewBox={`${viewBox.minX} ${viewBox.minY} ${viewBox.width} ${viewBox.height}`}
      >
        <g transform={`translate(${view.tx} ${view.ty}) scale(${view.scale})`}>
        <g className="graph-edges" aria-hidden="true">
          {edges.map((edge) => {
            const source = positionsById.get(edge.source);
            const target = positionsById.get(edge.target);
            if (!source || !target) return null;
            const connectedToSelection = selectedNodeId
              ? edge.source === selectedNodeId || edge.target === selectedNodeId
              : true;

            return (
              <line
                className={`graph-edge graph-edge-${edge.type} ${connectedToSelection ? 'is-related' : 'is-dimmed'}`}
                key={edge.id}
                x1={source.x}
                x2={target.x}
                y1={source.y}
                y2={target.y}
              />
            );
          })}
        </g>

        <g className="graph-nodes">
          {positionedNodes.map(({ node, x, y }) => {
            const isSelected = node.id === selectedNodeId;
            const isRelated = !selectedNodeId || relatedNodeIds.has(node.id);
            const selectNode = () => onSelectNode(node.id);

            return (
              <g
                aria-label={`Select ${node.label}`}
                aria-pressed={isSelected}
                className={`graph-node graph-node-${node.type} ${isSelected ? 'is-selected' : ''} ${
                  isRelated ? 'is-related' : 'is-dimmed'
                }`}
                key={node.id}
                onClick={selectNode}
                onKeyDown={(event) => handleControlKey(event, selectNode)}
                role="button"
                tabIndex={0}
                transform={`translate(${x - NODE_WIDTH / 2} ${y - NODE_HEIGHT / 2})`}
              >
                <title>{node.label}</title>
                <rect className="graph-node-box" height={NODE_HEIGHT} rx="0" width={NODE_WIDTH} />
                <text
                  className="graph-node-label"
                  textAnchor="middle"
                  x={NODE_WIDTH / 2}
                  y={showTypeLabel ? 19 : 28}
                >
                  {truncateLabel(node.label)}
                </text>
                {showTypeLabel ? (
                  <text className="graph-node-type" textAnchor="middle" x={NODE_WIDTH / 2} y={34}>
                    {node.type}
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>
        </g>
      </svg>
    </div>
  );
}
