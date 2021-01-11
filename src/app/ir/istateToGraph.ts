import { INodeState, IState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';
import Graph from '@/app/ir/Graph';
import { ModelNode } from '@/app/ir/mainNodes';
import { nodeBuilder } from '@/nodes/model/nodeBuilderMap';
import GraphNode from '@/app/ir/GraphNode';
import { UUID } from '@/app/util';
import { CommonNodes } from '@/nodes/common/Types';
import { CustomOptions } from '@/nodes/common/Custom';
import { OverviewNodes } from '@/nodes/overview/Types';
import ParsedFunction from '@/app/parser/ParsedFunction';

function traverseOptions(options: Array<[string, any]>): Map<string, any> {
  const constrMap: Map<string, any> = new Map<string, any>();
  for (const option of options) {
    constrMap.set(option[0], option[1]);
  }
  return constrMap;
}

function toGraphNode(inode: INodeState): ModelNode {
  const fromMap = nodeBuilder.get(inode.type);

  if (fromMap === undefined) {
    throw new Error(`${inode.type} is not mapped.`);
  }

  const options = traverseOptions(inode.options);
  if (inode.type === CommonNodes.Custom || inode.type === OverviewNodes.Custom) {
    const { parsedFunction } = inode.state;
    options.set(CustomOptions.Code, new ParsedFunction(parsedFunction.name,
      parsedFunction.body, parsedFunction.args, parsedFunction.filename).toString());
    options.set(CustomOptions.File, parsedFunction.filename);
  }

  options.set('name', inode.name);
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
  const nodesToDanglingInterfaces = new Map(
    istate.nodes.map((n) => [n.id, n.interfaces.filter(
      ([, i]) => !interfacesBackward.has(i.id) && !interfacesForward.has(i.id),
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
    // dangling interfaces
    for (const [interfaceName] of nodesToDanglingInterfaces.get(id)!) {
      graphNode.danglingInterfaces.push(interfaceName);
    }
  }

  // create connections
  const connections = istate.connections.map(
    (c) => [c.from, c.to].map((s) => new UUID(s)) as [UUID, UUID],
  );
  return new Graph(new Set(graphNodes.values()), connections);
}
