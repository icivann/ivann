import { randomUuid, UUID } from '@/app/util';

type MLNode = ModelNode

class GraphNode {
  constructor(
    public readonly mlNode: MLNode,
  ) {
  }

  public readonly uniqueId = randomUuid()
}

export abstract class ModelNode {
}

export type InOutNode = InNode & OutNode

export interface InNode {
  outputs: Set<UUID>;
}

export interface OutNode {
  input: UUID;
}

export interface Out2Node {
  input: [UUID, UUID]
}
