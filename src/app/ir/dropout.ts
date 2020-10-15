import { InOutNode, ModelNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

class Dropout implements ModelNode, InOutNode {
  constructor(
    public readonly input: UUID,
    public readonly outputs: Set<UUID>,

    public readonly probability: number,
  ) {}
}

export default Dropout;
