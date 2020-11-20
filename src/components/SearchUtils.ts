import { NodeListItem } from '@/components/canvas/AbstractCanvas';

interface SearchNode {
  name: string;
  displayName: string;
  options?: unknown;
  names?: Set<string>;
}

export interface SearchItem {
  category: string;
  nodes: SearchNode[];
}

export function convertToSearch(list: NodeListItem[]): SearchItem[] {
  const searchItems: SearchItem[] = [];
  for (const listItem of list) {
    const nodes: SearchNode[] = [];
    for (const node of listItem.nodes) {
      nodes.push({ name: node.name, displayName: node.name });
    }
    searchItems.push({
      category: listItem.category,
      nodes,
    });
  }
  return searchItems;
}
