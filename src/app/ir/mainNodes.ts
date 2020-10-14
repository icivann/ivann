import { randomUuid, UUID } from '@/app/util';

type MLNode = ModelNode | TrainingNode

class GraphNode {
  constructor(
    public readonly mlNode: MLNode,
  ) {
  }

  public readonly uniqueId = randomUuid()
}

export class ModelNode {
}

class TrainingNode {
}

export type InOutNode = InNode & OutNode

interface InNode {
  out: UUID;
}

interface OutNode {
  input: UUID;
}
