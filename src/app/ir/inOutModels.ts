import { InNode, InOutNode, OutNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

class InModel implements InNode {
  constructor(
    public readonly outputs: Set<UUID>,
    public readonly dimension: bigint[],
  ) {}
}

class OutModel implements OutNode {
  constructor(
    public readonly input: UUID,
  ) {}
}

class SoftMax implements InOutNode {
  constructor(
  public readonly input: UUID,
  public readonly outputs: Set<UUID>,
  ) {}
}
