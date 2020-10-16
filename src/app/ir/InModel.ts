import { UUID } from '@/app/util';
import { InNode } from '@/app/ir/mainNodes';

class InModel implements InNode {
  constructor(
    public readonly outputs: Set<UUID>,
    public readonly dimension: bigint[],
  ) {}
}

export default InModel;
