import { UUID } from '@/app/util';
import GraphNode from '@/app/ir/GraphNode';

export default class Graph {
  constructor(
    public readonly nodes: Set<GraphNode>,
    public readonly connections: Connection[],
  ) {
  }

  public readonly nodesByInputInterface: Map<UUID, GraphNode> = new Map(
    Array.from(this.nodes)
      .flatMap((n) => Array.from(n.inputInterfaces.values())
        .map((i) => [i, n] as [UUID, GraphNode])),
  )
}

type Connection = [UUID, UUID]
