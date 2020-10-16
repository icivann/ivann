import { UUID } from '@/app/util';
import { Padding } from '@/app/ir/irCommon';
import MaxPool from '@/app/ir/maxPool';

export default class MaxPool2D extends MaxPool {
  constructor(
        public readonly outputs: Set<UUID>,
        public readonly input: UUID,
        public padding: Padding,
        public readonly kernel: [bigint, bigint],
        public readonly stride: [bigint, bigint],
  ) {
    super();
  }

  public code(): string {
    const params = [
      `pool_size=(${this.kernel[0]}, ${this.kernel[1]})`,
      `strides=(${this.stride[0]}, ${this.stride[1]})`,
      `padding='${Padding[this.padding].toLowerCase()}'`,
    ];

    return `model.add(layers.MaxPool2D(${params.join(', ')}))`;
  }
}
