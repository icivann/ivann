import { IInterfaceState, IState } from '@baklavajs/core/dist/baklavajs-core/types';
import { ModelNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';
import GraphNode from '@/app/ir/GraphNode';
import Conv2D from '@/app/ir/Conv2D';

function mappingsForNodes(
  type: string,
  options: { [p: string]: any },
  interfacesMap: Map<string, ModelNode>,
  interfaces: Array<[string, IInterfaceState]>,
): ModelNode {
  const node = new Conv2D(
    new Set<UUID>(),
    options.Filters,
    options.Padding,
    [options['Kernel Size x'], options['Kernel Size y']],
    [options['Stride x'], options['Stride y']],
    new UUID('0'),
    options.Activation,
    [options['Weight Initializer'], options['Weight Regularizer']],
    [options['Bias Initializer'], options['Bias Regularizer']],
  );
  console.log(node.padding);
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
  [Set<GraphNode>, Array<[ModelNode| undefined, ModelNode | undefined]>] {
  const set: Set<GraphNode> = new Set();
  const interfacesMap: Map<string, ModelNode> = new Map<string, ModelNode>();
  const connections: Array<[ModelNode | undefined, ModelNode | undefined]> = new Array<[ModelNode, ModelNode]>();

  for (const node of state.nodes) {
    const constrMap: { [id: string]: any } = traverseOptions(node.options);
    const mlNode = mappingsForNodes(node.name, constrMap, interfacesMap, node.interfaces);
    const gNode = new GraphNode(mlNode, new UUID(node.id));
    set.add(gNode);
  }

  for (const connection of state.connections) {
    const fromNodeid = connection.from;
    const fromNode = interfacesMap.get(fromNodeid);
    console.log(fromNodeid);
    console.log(new UUID(fromNodeid));
    console.log(fromNode);
    const toNodeid = connection.to;
    const toNode = interfacesMap.get(toNodeid);
    console.log(connection);
    connections.push([fromNode, toNode]);
  }
  console.log(interfacesMap);
  console.log(set);
  console.log(connections);
  return [set, connections];
}
