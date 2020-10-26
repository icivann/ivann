import { IInterfaceState, IState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';
import { ModelNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';
import GraphNode from '@/app/ir/GraphNode';
import { nodeBuilder } from '@/nodes/model/nodeBuilderMap';

function mappingsForNodes(
  type: string,
  options: Map<string, any>,
  interfacesMap: Map<string, ModelNode>,
  interfaces: Array<[string, IInterfaceState]>,
): ModelNode {
  const fromMap = nodeBuilder.get(type);
  if (fromMap === undefined) {
    // TODO: throw exception?
    throw new Error(`${type} is not mapped.`);
  }
  const node = fromMap!(options);

  interfacesMap.set(interfaces[0][1].id, node);
  interfacesMap.set(interfaces[1][1].id, node);
  return node;
}

function traverseOptions(options: Array<[string, any]>): Map<string, any> {
  const constrMap: Map<string, any> = new Map<string, any>();
  for (const option of options) {
    constrMap.set(option[0], option[1]);
  }
  return constrMap;
}

export function traverseUiToIr(state: IState):
  [Set<GraphNode>, Array<[ModelNode, ModelNode]>] {
  const set: Set<GraphNode> = new Set();
  const interfacesMap: Map<string, ModelNode> = new Map<string, ModelNode>();
  const connections: Array<[ModelNode, ModelNode]> = new Array<[ModelNode, ModelNode]>();

  for (const node of state.nodes) {
    // TODO handle `CUSTOM` node case
    const constrMap: Map<string, any> = traverseOptions(node.options);
    const mlNode = mappingsForNodes(node.name, constrMap, interfacesMap, node.interfaces);
    const gNode = new GraphNode(mlNode, new UUID(node.id));
    set.add(gNode);
  }

  for (const connection of state.connections) {
    const fromNodeid = connection.from;
    const fromNode = interfacesMap.get(fromNodeid);

    const toNodeid = connection.to;
    const toNode = interfacesMap.get(toNodeid);
    if (fromNode && toNode) {
      connections.push([fromNode, toNode]);
    }
  }

  console.log(set);
  console.log(connections);
  return [set, connections];
}
