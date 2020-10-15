import { InOutNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

class Softmax implements InOutNode {
  constructor(
  public readonly input: UUID,
  public readonly outputs: Set<UUID>,
  ) {}
}

export default Softmax;
