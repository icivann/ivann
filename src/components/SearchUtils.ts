import { NodeListItem } from '@/components/canvas/AbstractCanvas';

export interface SearchNode {
  name: string;
  displayName: string;
  options?: unknown;
}

export interface SearchItem {
  category: string;
  nodes: SearchNode[];
}

export function convertToSearch(list: NodeListItem[]): SearchItem[] {
  return list.map((listItem) => {
    const nodes = listItem.nodes.map((node) => ({ ...node, displayName: node.name }));
    return { category: listItem.category, nodes };
  });
}

export function modify(list: SearchItem[], category: string, name: string, newValue: SearchNode) {
  return list.map((searchItem) => {
    if (searchItem.category === category) {
      return {
        category,
        nodes: searchItem.nodes.map((node) => {
          if (node.name === name) {
            return newValue;
          }
          return node;
        }),
      };
    }
    return searchItem;
  });
}

export function search(list: SearchItem[], searchString: string) {
  const search = searchString.toLowerCase();
  const result: SearchItem[] = [];
  for (const category of list) {
    const nodes = category.nodes.filter((node) => node
      .displayName.toLowerCase().includes(search));
    if (nodes.length > 0) result.push({ category: category.category, nodes });
  }
  return result;
}
