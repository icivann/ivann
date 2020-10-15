import { OutNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

class OutModel implements OutNode {
  constructor(
    public readonly input: UUID,
  ) {}
}

export default OutModel;
