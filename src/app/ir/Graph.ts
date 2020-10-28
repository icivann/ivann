import { UUID } from '@/app/util';
import GraphNode from '@/app/ir/GraphNode';

export default class Graph {
  constructor(
    public readonly nodes: Set<GraphNode>,
    public readonly connections: Connection[],
  ) {
  }

  public readonly inToOutConnections = new Map(this.connections)
  public readonly nodesAsArray = Array.from(this.nodes)

  public readonly nodesByInputInterface: Map<UUID, GraphNode> = new Map(
    this.nodesAsArray
      .flatMap((n) => Array.from(n.inputInterfaces.values())
        .map((i) => [i, n] as [UUID, GraphNode])),
  )

  public readonly nodesFromNode: Map<GraphNode, GraphNode[]> = new Map(
    this.nodesAsArray
      .map((sourceNode) => {
        const nodes = Array.from(sourceNode.outputInterfaces.values())
          .map((i) => this.nodesByInputInterface.get(this.inToOutConnections.get(i)!)!);
        return [sourceNode, nodes];
      }),
  )

  nextNodesFrom(source: GraphNode): GraphNode[] {
    return this.nodesFromNode.get(source)!;
  }
}

type Connection = [UUID, UUID]
