import GraphNode from '@/app/ir/GraphNode';

export const indent = '  ';

export function getNodeType(node: GraphNode): string {
  return node.mlNode.constructor.name.toLowerCase();
}
