import { INodeState, IState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';
import Graph from '@/app/ir/Graph';
import { ModelNode } from '@/app/ir/mainNodes';
import { nodeBuilder } from '@/nodes/model/nodeBuilderMap';
import GraphNode from '@/app/ir/GraphNode';
import { UUID } from '@/app/util';

function traverseOptions(options: Array<[string, any]>): Map<string, any> {
  const constrMap: Map<string, any> = new Map<string, any>();
  for (const option of options) {
    constrMap.set(option[0], option[1]);
  }
  return constrMap;
}

function toGraphNode(inode: INodeState): ModelNode {
  const fromMap = nodeBuilder.get(inode.name);
  if (fromMap === undefined) {
    // TODO: throw exception?
    throw new Error(`${inode.name} is not mapped.`);
  }
  const options = traverseOptions(inode.options);
  return fromMap!(options);
}

export default function istateToGraph(istate: IState): Graph {
  const interfacesForward = new Map(istate.connections.map(
    (c) => [c.from, c.to],
  ));
  const interfacesBackward = new Map(istate.connections.map(
    (c) => [c.to, c.from],
  ));
  const graphNodes = new Map(
    istate.nodes.map(
      (n) => [n.id, new GraphNode(toGraphNode(n), new UUID(n.id))],
    ),
  );
  const nodesToOutInterfaces = new Map(
    istate.nodes.map((n) => [n.id, n.interfaces.filter(
      ([_, i]) => interfacesForward.has(i.id),
    )]),
  );
  const nodesToInInterfaces = new Map(
    istate.nodes.map((n) => [n.id, n.interfaces.filter(
      ([_, i]) => interfacesBackward.has(i.id),
    )]),
  );

  // add interfaces to individual nodes
  for (const [id, graphNode] of graphNodes) {
    for (const [interfaceName, interfaceI] of nodesToInInterfaces.get(id)!) {
      graphNode.inputInterfaces.set(interfaceName, new UUID(interfaceI.id));
    }
    for (const [interfaceName, interfaceI] of nodesToOutInterfaces.get(id)!) {
      graphNode.outputInterfaces.set(interfaceName, new UUID(interfaceI.id));
    }
  }

  // create connections
  const connections = istate.connections.map(
    (c) => [c.from, c.to].map((s) => new UUID(s)) as [UUID, UUID],
  );
  console.log(graphNodes.values());
  console.log(istate);
  return new Graph(new Set(graphNodes.values()), connections);
}
