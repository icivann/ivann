import { randomUuid } from '@/app/util';
import { ModelNode } from '@/app/ir/mainNodes';

export default class GraphNode {
  constructor(
    public readonly modelNode: ModelNode,
    public readonly uniqueId = randomUuid(),
  ) {
  }
}
