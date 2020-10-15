import { InNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

class InModel implements InNode {
  constructor(
    public readonly outputs: Set<UUID>,
    public readonly dimension: bigint[],
  ) {}
  public code(): string {
    const input_shape = `(${this.dimension.join(', ')})`;
    return `model.add(keras.Input(shape=${input_shape}))\n`;
  }
}

export default InModel;
