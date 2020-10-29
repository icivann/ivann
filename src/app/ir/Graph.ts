import { UUID } from '@/app/util';
import GraphNode from '@/app/ir/GraphNode';

export default class Graph {
  constructor(
    public readonly nodes: Set<GraphNode>,
    public readonly connections: Connection[],
  ) {
  }

  public readonly inToOutConnections = new Map(this.connections.map(([i, o]) => [i.id, o.id]))
  public readonly nodesAsArray = [...this.nodes]

  public readonly nodesByInputInterface: Map<string, GraphNode> = new Map(
    this.nodesAsArray
      .flatMap((n) => Array.from(n.inputInterfaces.values())
        .map((i) => [i.id, n] as [string, GraphNode])),
  )

  public readonly nodesFromNode: Map<GraphNode, GraphNode[]> = new Map(
    this.nodesAsArray
      .map((sourceNode) => {
        const nodes = Array.from(sourceNode.outputInterfaces.values())
          .map((i) => this.nodesByInputInterface.get(this.inToOutConnections.get(i.id)!)!);
        return [sourceNode, nodes];
      }),
  )

  nextNodesFrom(source: GraphNode): GraphNode[] {
    return this.nodesFromNode.get(source)!;
  }

  nextNodeFrom(source: GraphNode, iName: string): GraphNode | undefined {
    const next = source.outputInterfaces.get(iName);
    return next === undefined
      ? undefined
      : this.nodesByInputInterface.get(this.inToOutConnections.get(next.id)!);
  }
}

type Connection = [UUID, UUID]
