import { posix } from 'node:path';
import { stableId } from './path-utils.js';
import type { DocumentKind, WikiDocument, WikiIndex } from './types.js';

export type GraphNodeType = 'document' | 'topic' | 'tag' | 'source';
export type GraphEdgeType = 'belongs_to_topic' | 'has_tag' | 'links_to' | 'cites_source' | 'same_tag';

export interface GraphNode {
  id: string;
  type: GraphNodeType;
  label: string;
  topic?: string;
  documentId?: string;
  documentKind?: DocumentKind;
  archived?: boolean;
  summary?: string;
  weight: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: GraphEdgeType;
  weight: number;
  label?: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    nodeCount: number;
    edgeCount: number;
    omittedNodeCount: number;
    omittedEdgeCount: number;
  };
}

export interface GraphQuery {
  includeArchived?: boolean;
  topic?: string;
  documentId?: string;
  depth?: number;
  nodeTypes?: GraphNodeType[];
  edgeTypes?: GraphEdgeType[];
  maxNodes?: number;
  maxEdges?: number;
}

interface MutableNode extends GraphNode {
  weight: number;
}

const DEFAULT_MAX_NODES = 250;
const DEFAULT_MAX_EDGES = 500;
const SAME_TAG_EDGE_LIMIT_PER_TAG = 25;

const NODE_TYPE_ORDER: Record<GraphNodeType, number> = {
  document: 0,
  topic: 1,
  tag: 2,
  source: 3,
};

const EDGE_TYPE_ORDER: Record<GraphEdgeType, number> = {
  belongs_to_topic: 0,
  has_tag: 1,
  links_to: 2,
  cites_source: 3,
  same_tag: 4,
};

function cleanLimit(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value)) return fallback;
  return Math.max(0, Math.floor(value));
}

function cleanDepth(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) return 1;
  return Math.max(0, Math.floor(value));
}

function normalizeSlashes(value: string): string {
  return value.replaceAll('\\', '/');
}

function normalizeTag(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9一-鿿]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function normalizeLookup(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9一-鿿]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function normalizedRelativePath(value: string): string {
  return posix.normalize(normalizeSlashes(value)).replace(/^\.\//, '');
}

function withoutMarkdownExtension(value: string): string {
  return value.replace(/\.md$/i, '');
}

function stripAnchor(value: string): string {
  return value.split('#')[0] ?? value;
}

function firstWikiLinkTarget(rawTarget: string): string {
  return stripAnchor(rawTarget.split('|')[0]?.trim() ?? '');
}

function isRemoteReference(value: string): boolean {
  return /^[a-z][a-z0-9+.-]*:/i.test(value);
}

function looksLikeLocalReference(value: string): boolean {
  return (
    value.startsWith('/') ||
    value.startsWith('./') ||
    value.startsWith('../') ||
    value.includes('\\') ||
    value.includes('/') ||
    /\.md(?:#|$)/i.test(value)
  );
}

function sourceKey(value: string): string {
  const trimmed = value.trim();
  try {
    const url = new URL(trimmed);
    url.hash = '';
    return url.toString();
  } catch {
    return trimmed;
  }
}

function sourceLabel(value: string): string {
  const key = sourceKey(value);
  try {
    return new URL(key).hostname || 'Source';
  } catch {
    if (looksLikeLocalReference(key)) return 'Local source';
    return key.slice(0, 80) || 'Source';
  }
}

function sourceValues(source: unknown): string[] {
  if (typeof source === 'string') return source.trim() ? [source.trim()] : [];
  if (Array.isArray(source)) {
    return source.flatMap((entry) => sourceValues(entry));
  }
  if (source && typeof source === 'object') {
    const candidate = source as Record<string, unknown>;
    return sourceValues(candidate.url ?? candidate.uri ?? candidate.href ?? candidate.path ?? candidate.source);
  }
  return [];
}

function edgeId(type: GraphEdgeType, source: string, target: string, label?: string): string {
  return `edge:${stableId(['graph-edge', type, source, target, label ?? ''])}`;
}

function addNode(nodes: Map<string, MutableNode>, node: GraphNode): void {
  const existing = nodes.get(node.id);
  if (!existing) {
    nodes.set(node.id, { ...node });
    return;
  }

  existing.weight = Math.max(existing.weight, node.weight);
  existing.summary ??= node.summary;
  existing.archived = Boolean(existing.archived || node.archived);
}

function addEdge(edges: Map<string, GraphEdge>, edge: Omit<GraphEdge, 'id'>): void {
  const id = edgeId(edge.type, edge.source, edge.target, edge.label);
  if (!edges.has(id)) edges.set(id, { id, ...edge });
}

function indexDocumentsByLinkTarget(documents: WikiDocument[]): Map<string, string> {
  const targets = new Map<string, string>();

  for (const document of documents) {
    const candidates = [
      document.id,
      document.title,
      posix.basename(document.relativePath, '.md'),
      withoutMarkdownExtension(document.relativePath),
      withoutMarkdownExtension(normalizedRelativePath(document.relativePath)),
    ];

    for (const candidate of candidates) {
      const key = normalizeLookup(candidate);
      if (key && !targets.has(key)) targets.set(key, document.id);
    }
  }

  return targets;
}

function markdownPathTarget(document: WikiDocument, rawHref: string, documentsByPath: Map<string, string>): string | undefined {
  const href = stripAnchor(rawHref.trim().replace(/^<|>$/g, ''));
  if (!href || isRemoteReference(href) || href.startsWith('#')) return undefined;

  let decoded = href;
  try {
    decoded = decodeURIComponent(href);
  } catch {
    decoded = href;
  }

  const normalizedHref = normalizeSlashes(decoded);
  const currentDir = posix.dirname(normalizedRelativePath(document.relativePath));
  const resolved = normalizedHref.startsWith('/')
    ? normalizedRelativePath(normalizedHref.slice(1))
    : normalizedRelativePath(posix.join(currentDir, normalizedHref));

  return documentsByPath.get(resolved) ?? documentsByPath.get(withoutMarkdownExtension(resolved));
}

function extractDocumentLinks(
  document: WikiDocument,
  documentsByLinkTarget: Map<string, string>,
  documentsByPath: Map<string, string>,
): string[] {
  const targets = new Set<string>();

  for (const match of document.body.matchAll(/\[\[([^\]\n]+)\]\]/g)) {
    const rawTarget = firstWikiLinkTarget(match[1] ?? '');
    const target = documentsByLinkTarget.get(normalizeLookup(rawTarget));
    if (target && target !== document.id) targets.add(target);
  }

  for (const match of document.body.matchAll(/(?<!!)\[[^\]\n]+\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/g)) {
    const target = markdownPathTarget(document, match[1] ?? '', documentsByPath);
    if (target && target !== document.id) targets.add(target);
  }

  return [...targets].sort((left, right) => left.localeCompare(right));
}

function fullGraph(index: WikiIndex, query: GraphQuery): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const includeArchived = Boolean(query.includeArchived);
  const visibleTopics = index.topics
    .filter((topic) => includeArchived || !topic.archived)
    .filter((topic) => !query.topic || topic.slug === query.topic);
  const visibleTopicIds = new Set(visibleTopics.map((topic) => topic.slug));
  const indexedTopicIds = new Set(index.topics.map((topic) => topic.slug));
  const documents = index.documents
    .filter((document) => includeArchived || !document.archived)
    .filter((document) => !query.topic || document.topic === query.topic)
    .filter((document) => {
      if (query.topic || indexedTopicIds.has(document.topic)) return visibleTopicIds.has(document.topic);
      return true;
    })
    .sort((left, right) => left.id.localeCompare(right.id));
  const documentIds = new Set(documents.map((document) => document.id));

  const nodes = new Map<string, MutableNode>();
  const edges = new Map<string, GraphEdge>();
  const tagDocs = new Map<string, Set<string>>();
  const tagLabels = new Map<string, string>();
  const sourceDocs = new Map<string, Set<string>>();
  const documentsByLinkTarget = indexDocumentsByLinkTarget(documents);
  const documentsByPath = new Map<string, string>();

  for (const document of documents) {
    const path = normalizedRelativePath(document.relativePath);
    documentsByPath.set(path, document.id);
    documentsByPath.set(withoutMarkdownExtension(path), document.id);
  }

  for (const topic of visibleTopics.sort((left, right) => left.slug.localeCompare(right.slug))) {
    const topicDocs = documents.filter((document) => document.topic === topic.slug);
    if (topicDocs.length === 0 && !includeArchived) continue;
    addNode(nodes, {
      id: `topic:${topic.slug}`,
      type: 'topic',
      label: topic.slug,
      topic: topic.slug,
      archived: topic.archived,
      summary: topic.description,
      weight: Math.max(1, topicDocs.length),
    });
  }

  for (const document of documents) {
    const links = extractDocumentLinks(document, documentsByLinkTarget, documentsByPath);

    addNode(nodes, {
      id: document.id,
      type: 'document',
      label: document.title,
      topic: document.topic,
      documentId: document.id,
      documentKind: document.kind,
      archived: document.archived,
      summary: document.summary,
      weight: Math.max(1, 1 + document.tags.length + sourceValues(document.source).length + links.length),
    });

    const topicNodeId = `topic:${document.topic}`;
    if (nodes.has(topicNodeId)) {
      addEdge(edges, { source: document.id, target: topicNodeId, type: 'belongs_to_topic', weight: 1 });
    }

    for (const tag of document.tags) {
      const normalized = normalizeTag(tag);
      if (!normalized) continue;
      tagLabels.set(normalized, tagLabels.get(normalized) ?? tag.trim());
      const tagNodeId = `tag:${normalized}`;
      const taggedDocuments = tagDocs.get(normalized) ?? new Set<string>();
      taggedDocuments.add(document.id);
      tagDocs.set(normalized, taggedDocuments);
      addNode(nodes, {
        id: tagNodeId,
        type: 'tag',
        label: tagLabels.get(normalized) ?? normalized,
        topic: document.topic,
        weight: taggedDocuments.size,
      });
      addEdge(edges, { source: document.id, target: tagNodeId, type: 'has_tag', weight: 1 });
    }

    for (const value of sourceValues(document.source)) {
      const key = sourceKey(value);
      if (!key) continue;
      const sourceNodeId = `source:${stableId(['graph-source', key])}`;
      const citingDocuments = sourceDocs.get(sourceNodeId) ?? new Set<string>();
      citingDocuments.add(document.id);
      sourceDocs.set(sourceNodeId, citingDocuments);
      addNode(nodes, {
        id: sourceNodeId,
        type: 'source',
        label: sourceLabel(value),
        weight: citingDocuments.size,
      });
      addEdge(edges, { source: document.id, target: sourceNodeId, type: 'cites_source', weight: 1 });
    }

    for (const target of links) {
      if (documentIds.has(target)) addEdge(edges, { source: document.id, target, type: 'links_to', weight: 1 });
    }
  }

  for (const [tag, taggedDocuments] of [...tagDocs.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    const ids = [...taggedDocuments].sort((left, right) => left.localeCompare(right));
    let emitted = 0;
    for (let leftIndex = 0; leftIndex < ids.length && emitted < SAME_TAG_EDGE_LIMIT_PER_TAG; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < ids.length && emitted < SAME_TAG_EDGE_LIMIT_PER_TAG; rightIndex += 1) {
        addEdge(edges, {
          source: ids[leftIndex],
          target: ids[rightIndex],
          type: 'same_tag',
          weight: 1,
          label: tagLabels.get(tag) ?? tag,
        });
        emitted += 1;
      }
    }
  }

  return { nodes: sortNodes([...nodes.values()]), edges: sortEdges([...edges.values()]) };
}

function sortNodes(nodes: GraphNode[]): GraphNode[] {
  return nodes.sort((left, right) => NODE_TYPE_ORDER[left.type] - NODE_TYPE_ORDER[right.type] || left.id.localeCompare(right.id));
}

function sortEdges(edges: GraphEdge[]): GraphEdge[] {
  return edges.sort(
    (left, right) =>
      EDGE_TYPE_ORDER[left.type] - EDGE_TYPE_ORDER[right.type] ||
      left.source.localeCompare(right.source) ||
      left.target.localeCompare(right.target) ||
      left.id.localeCompare(right.id),
  );
}

function documentNeighborhood(nodes: GraphNode[], edges: GraphEdge[], documentId: string, depth: number): { nodes: GraphNode[]; edges: GraphEdge[] } {
  if (!nodes.some((node) => node.id === documentId)) return { nodes: [], edges: [] };

  const adjacency = new Map<string, Set<string>>();
  for (const edge of edges) {
    const sourceNeighbors = adjacency.get(edge.source) ?? new Set<string>();
    const targetNeighbors = adjacency.get(edge.target) ?? new Set<string>();
    sourceNeighbors.add(edge.target);
    targetNeighbors.add(edge.source);
    adjacency.set(edge.source, sourceNeighbors);
    adjacency.set(edge.target, targetNeighbors);
  }

  const distances = new Map<string, number>([[documentId, 0]]);
  const queue = [documentId];
  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index];
    const currentDistance = distances.get(current) ?? 0;
    if (currentDistance >= depth) continue;

    for (const neighbor of [...(adjacency.get(current) ?? [])].sort((left, right) => left.localeCompare(right))) {
      if (distances.has(neighbor)) continue;
      distances.set(neighbor, currentDistance + 1);
      queue.push(neighbor);
    }
  }

  const keptNodes = nodes.filter((node) => distances.has(node.id));
  const keptEdges = edges.filter((edge) => {
    const sourceDistance = distances.get(edge.source);
    const targetDistance = distances.get(edge.target);
    if (sourceDistance === undefined || targetDistance === undefined) return false;
    return Math.min(sourceDistance, targetDistance) < depth && Math.max(sourceDistance, targetDistance) <= depth;
  });

  return { nodes: keptNodes, edges: keptEdges };
}

function applyTypeFilters(nodes: GraphNode[], edges: GraphEdge[], query: GraphQuery): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const allowedNodeTypes = query.nodeTypes ? new Set(query.nodeTypes) : undefined;
  const allowedEdgeTypes = query.edgeTypes ? new Set(query.edgeTypes) : undefined;
  const filteredNodes = allowedNodeTypes ? nodes.filter((node) => allowedNodeTypes.has(node.type)) : nodes;
  const nodeIds = new Set(filteredNodes.map((node) => node.id));
  const filteredEdges = edges.filter(
    (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target) && (!allowedEdgeTypes || allowedEdgeTypes.has(edge.type)),
  );

  return { nodes: filteredNodes, edges: filteredEdges };
}

function applyCaps(nodes: GraphNode[], edges: GraphEdge[], query: GraphQuery): GraphResponse {
  const maxNodes = cleanLimit(query.maxNodes, DEFAULT_MAX_NODES);
  const maxEdges = cleanLimit(query.maxEdges, DEFAULT_MAX_EDGES);
  const pinnedNode = query.documentId && maxNodes > 0 ? nodes.find((node) => node.id === query.documentId) : undefined;
  const returnedNodes = pinnedNode
    ? [pinnedNode, ...nodes.filter((node) => node.id !== pinnedNode.id).slice(0, maxNodes - 1)]
    : nodes.slice(0, maxNodes);
  const returnedNodeIds = new Set(returnedNodes.map((node) => node.id));
  const edgesWithReturnedNodes = edges.filter((edge) => returnedNodeIds.has(edge.source) && returnedNodeIds.has(edge.target));
  const returnedEdges = edgesWithReturnedNodes.slice(0, maxEdges);

  return {
    nodes: returnedNodes,
    edges: returnedEdges,
    stats: {
      nodeCount: returnedNodes.length,
      edgeCount: returnedEdges.length,
      omittedNodeCount: nodes.length - returnedNodes.length,
      omittedEdgeCount: edges.length - returnedEdges.length,
    },
  };
}

export function buildKnowledgeGraph(index: WikiIndex, query: GraphQuery = {}): GraphResponse {
  let graph = fullGraph(index, query);

  if (query.documentId) {
    graph = documentNeighborhood(graph.nodes, graph.edges, query.documentId, cleanDepth(query.depth));
  }

  graph = applyTypeFilters(graph.nodes, graph.edges, query);
  return applyCaps(sortNodes(graph.nodes), sortEdges(graph.edges), query);
}
