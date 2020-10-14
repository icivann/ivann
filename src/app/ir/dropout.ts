import { InOutNode, ModelNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

class Dropout implements ModelNode, InOutNode {
  constructor(
    public readonly input: UUID,
    public readonly out: UUID,

    public readonly probability: 0.5,
  ) {}
}
