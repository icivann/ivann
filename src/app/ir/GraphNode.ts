import { randomUuid } from '@/app/util';
import { MlNode } from '@/app/ir/mainNodes';

class GraphNode {
  constructor(
    public readonly mlNode: MlNode,
  ) {
  }

  public readonly uniqueId = randomUuid()
}
