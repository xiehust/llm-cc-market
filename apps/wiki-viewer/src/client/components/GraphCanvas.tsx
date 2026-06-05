import { useMemo } from 'react';
import type { KeyboardEvent } from 'react';
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

const WIDTH = 960;
const HEIGHT = 620;
const CENTER_X = WIDTH / 2;
const CENTER_Y = HEIGHT / 2;
const NODE_WIDTH = 150;
const NODE_HEIGHT = 58;

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

function layoutNodes(nodes: GraphNodeDto[]): PositionedNode[] {
  const ordered = sortedNodes(nodes);
  if (ordered.length === 0) return [];
  if (ordered.length === 1) return [{ node: ordered[0], x: CENTER_X, y: CENTER_Y }];

  const radiusX = Math.min(330, 165 + ordered.length * 18);
  const radiusY = Math.min(220, 112 + ordered.length * 12);
  const startAngle = -Math.PI / 2;

  return ordered.map((node, index) => {
    const angle = startAngle + (index / ordered.length) * Math.PI * 2;
    return {
      node,
      x: CENTER_X + Math.cos(angle) * radiusX,
      y: CENTER_Y + Math.sin(angle) * radiusY,
    };
  });
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
  return label.length > 24 ? `${label.slice(0, 21)}...` : label;
}

function handleControlKey(event: KeyboardEvent<SVGGElement>, onSelect: () => void): void {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  event.preventDefault();
  onSelect();
}

export default function GraphCanvas({ nodes, edges, selectedNodeId, onSelectNode }: GraphCanvasProps) {
  const positionedNodes = useMemo(() => layoutNodes(nodes), [nodes]);
  const positionsById = useMemo(() => {
    return new Map(positionedNodes.map((entry) => [entry.node.id, entry]));
  }, [positionedNodes]);
  const relatedNodeIds = useMemo(() => immediateNeighborIds(edges, selectedNodeId), [edges, selectedNodeId]);

  return (
    <div className="graph-canvas-frame">
      <svg aria-label="Knowledge graph" className="graph-canvas" role="group" viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
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
                <text className="graph-node-label" x="12" y="25">
                  {truncateLabel(node.label)}
                </text>
                <text className="graph-node-type" x="12" y="45">
                  {node.type}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
