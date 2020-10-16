import { InNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

export class InModel implements InNode {
  constructor(
    public readonly outputs: Set<UUID>,
    public readonly dimension: bigint[],
  ) {}

  public code(): string {
    const inputShape = `(${this.dimension.join(', ')})`;
    return `model.add(keras.Input(shape=${inputShape}))\n`;
  }
}

export default InModel;
