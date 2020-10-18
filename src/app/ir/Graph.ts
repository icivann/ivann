import { UUID } from '@/app/util';
import GraphNode from '@/app/ir/GraphNode';

export default class Graph {
  constructor(
        public readonly nodes: Map<UUID, GraphNode>,
        public readonly connections: Connection[],
  ) {
  }
}

type Connection = [UUID, UUID]
