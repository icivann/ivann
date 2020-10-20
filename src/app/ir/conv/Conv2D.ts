import { Option } from '@/app/util';
import {
  ActivationF, BuiltinActivationF, Initializer, Padding, Regularizer,
} from '@/app/ir/irCommon';

export default class Conv2D {
  constructor(
    public readonly filters: bigint,
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly activation: ActivationF,
    public readonly kernel: [bigint, bigint],
    public readonly stride: [bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): Conv2D {
    console.log(options);
    const node = new Conv2D(
      options.get('Filters'),
      options.get('Padding'),
      [options.get('Weights Initializer'), options.get('Weights Regularizer')],
      [options.get('Bias Initializer'), options.get('Bias Regularizer')],
      BuiltinActivationF[options.get('Activation') as keyof typeof BuiltinActivationF],
      [options.get('Kernel Size')[0], options.get('Kernel Size')[1]],
      [options.get('Stride')[0], options.get('Stride')[1]],
    );
    return node;
  }
}
