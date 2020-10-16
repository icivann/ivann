import { OutNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

class OutModel implements OutNode {
  constructor(
    public readonly input: UUID,
  ) {}
  public code(): string {
    const a = this.input;
    return `${a}`;
  }
}

export default OutModel;
