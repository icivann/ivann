import Conv from '@/app/ir/Conv';
import { Option, UUID } from '@/app/util';
import {
  ActivationF, BuiltinActivationF, Initializer, Padding, Regularizer,
} from '@/app/ir/irCommon';

export default class Conv2D extends Conv {
  constructor(
    public readonly outputs: Set<UUID>,
    public readonly filters: bigint,
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly input: UUID,
    public readonly activation: ActivationF,

    public readonly kernel: [bigint, bigint],
    public readonly stride: [bigint, bigint],
  ) {
    super();
  }

  public code(): string {
    const params = [
      `${this.filters}`,
      `(${this.kernel[0]}, ${this.kernel[1]}) `,
      `strides = (${this.stride[0]}, ${this.stride[1]}) `,
      `padding = ${Padding[this.padding]}`,
      `activation = ${BuiltinActivationF[this.activation]}`,
      `use_bias = ${this}`,
      kernel_initializer='glorot_uniform', bias_initializer='zeros',
      kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
      kernel_constraint=None, bias_constraint=None
    ];

    if (this.biases !== null) {
      const initializer = this.biases[0];
      const regularizer = this.biases[1];
    }

    return 'model.add(layers.Conv2D())';
  }
}
