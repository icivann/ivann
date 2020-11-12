import { randomUuid, UUID } from '@/app/util';
import { MlNode } from '@/app/ir/mainNodes';

export default class GraphNode {
  constructor(
    public readonly mlNode: MlNode,
    public readonly uniqueId = randomUuid(),
    public readonly inputInterfaces = new Map<string, UUID>(),
    public readonly outputInterfaces = new Map<string, UUID>(),
    public readonly danglingInterfaces = new Array<string>(),
  ) {
  }
}
