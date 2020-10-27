import {
  ActivationF, getBuiltinActivationFunction,
  getInitializer,
  getRegularizer,
  Initializer,
  Regularizer
} from '@/app/ir/irCommon';
import { Option } from '@/app/util';
import { DenseOptions } from '@/nodes/model/linear/Dense.ts';

export default class Dense {
  constructor(
    public readonly size: bigint,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly activation: ActivationF,
  ) {
  }

  static build(options: Map<string, any>): Dense {
    return new Dense(
      options.get(DenseOptions.Size),
      [getInitializer(options.get(DenseOptions.WeightsInitializer)),
        getRegularizer(options.get(DenseOptions.WeightsRegularizer))],
      [getInitializer(options.get(DenseOptions.BiasInitializer)),
        getRegularizer(options.get(DenseOptions.BiasRegularizer))],
      getBuiltinActivationFunction(options.get(DenseOptions.Activation)),
    );
  }
}
