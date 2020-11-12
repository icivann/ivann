import { UUID } from '@/app/util';
import GraphNode from '@/app/ir/GraphNode';

export default class Graph {
  constructor(
    public readonly nodes: Set<GraphNode>,
    public readonly connections: Connection[],
  ) {
  }

  public readonly inToOutConnections = new Map((() => {
    const inConnections = new Set(this.connections.map(([i, _]) => i));
    return Array.from(inConnections).map(
      (i) => {
        const outputs = this.connections
          .filter(([input, o]) => i.id === input.id)
          .map((c) => c[1]);
        return [i.id, outputs] as [string, UUID[]];
      },
    );
  })())
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
          .flatMap((i) => this.inToOutConnections.get(i.id)!
            .map((c) => this.nodesByInputInterface.get(c.id)!));
        return [sourceNode, nodes];
      }),
  )

  nextNodesFrom(source: GraphNode): GraphNode[] {
    return this.nodesFromNode.get(source)!;
  }

  nextNodeFrom(source: GraphNode, iName: string): GraphNode[] | undefined {
    const next = source.outputInterfaces.get(iName);
    return next === undefined
      ? undefined
      : this.inToOutConnections.get(next.id)?.map((c) => this.nodesByInputInterface.get(c.id)!);
  }
}

export type Connection = [UUID, UUID]
