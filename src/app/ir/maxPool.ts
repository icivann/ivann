import { InOutNode, ModelNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

abstract class MaxPool implements ModelNode, InOutNode {
  abstract readonly outputs: Set<UUID>;

  abstract readonly input: UUID;

  public abstract code(): string;
}

export default MaxPool;
