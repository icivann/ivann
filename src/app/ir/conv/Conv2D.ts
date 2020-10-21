import { Option } from '@/app/util';
import {
  ActivationF,
  getBuiltinActivationFunction,
  getInitializer,
  getRegularizer,
  Initializer,
  Padding,
  Regularizer,
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
    return new Conv2D(
      options.get('Filters'),
      options.get('Padding'),
      [getInitializer(options.get('Weights Initializer')), getRegularizer(options.get('Weights Regularizer'))],
      [getInitializer(options.get('Bias Initializer')), getRegularizer(options.get('Bias Regularizer'))],
      getBuiltinActivationFunction(options.get('Activation')),
      [options.get('Kernel Size')[0], options.get('Kernel Size')[1]],
      [options.get('Stride')[0], options.get('Stride')[1]],
    );
  }
}
