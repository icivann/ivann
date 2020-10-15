import { InNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

class InModel implements InNode {
  constructor(
    public readonly outputs: Set<UUID>,
    public readonly dimension: bigint[],
  ) {}
}

export default InModel;
