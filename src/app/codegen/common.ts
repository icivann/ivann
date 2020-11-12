import GraphNode from '@/app/ir/GraphNode';

export const indent = '  ';

export function getNodeType(node: GraphNode): string {
  return node.mlNode.name.toLowerCase();
}
