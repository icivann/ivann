import { randomUuid, UUID } from '@/app/util';
import { MlNode } from '@/app/ir/mainNodes';

export default class GraphNode {
  constructor(
    public readonly modelNode: MlNode,
    public readonly uniqueId = randomUuid(),
    public readonly inputInterfaces =
    new Map<string, UUID>(),
    public readonly outputInterfaces =
    new Map<string, UUID>(),
  ) {
  }
}
