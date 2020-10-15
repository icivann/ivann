import Conv from '@/app/ir/Conv';
import { Option, UUID } from '@/app/util';
import {
  ActivationF, BuiltinActivationF, BuiltinInitializer, BuiltinRegularizer, Initializer,
  Padding, Regularizer,
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
    const weightsInitializer = this.weights[0] as BuiltinInitializer;
    const weightsRegularizer = this.weights[1] as BuiltinRegularizer;

    const params = [
      `${this.filters}`,
      `(${this.kernel[0]}, ${this.kernel[1]}) `,
      `strides = (${this.stride[0]}, ${this.stride[1]}) `,
      `padding = ${Padding[this.padding]}`,
      `activation = ${BuiltinActivationF[this.activation]}`,
      // TODO: fix assumption that all initializers and regularizers are built in types
      `kernel_initializer = ${BuiltinInitializer[weightsInitializer].toLowerCase()}`,
      `kernel_regularizer = ${BuiltinRegularizer[weightsRegularizer]}`,
    ];
    params.push(`use_bias = ${this.biases !== null}`);

    if (this.biases !== null) {
      // TODO: fix assumption that all initializers and regularizers are built in types
      const initializer = this.biases[0] as BuiltinInitializer;
      const regularizer = this.biases[1] as BuiltinRegularizer;

      params.push(`bias_initializer = ${BuiltinInitializer[initializer].toLowerCase()}`);
      params.push(`bias_regularizer = ${BuiltinRegularizer[regularizer]}`);
    }

    return `model.add(layers.Conv2D(${params.join(',')}))`;
  }
}
