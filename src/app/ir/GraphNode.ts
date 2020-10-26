import { randomUuid } from '@/app/util';
import { MlNode } from '@/app/ir/mainNodes';

export default class GraphNode {
  constructor(
    public readonly mlNode: MlNode,
    public readonly uniqueId = randomUuid(),
    public readonly inputInterfaces =
    new Map([['input', randomUuid()]]),
    public readonly outputInterfaces =
    new Map([['output', randomUuid()]]),
  ) {
  }
}
