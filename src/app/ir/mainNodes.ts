import { randomUuid, UUID } from '@/app/util';

type MlNode = ModelNode

class GraphNode {
  constructor(
    public readonly mlNode: MlNode,
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
