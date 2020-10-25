import { Option } from '@/app/util';
import {
  ActivationF,
  getBuiltinActivationFunction,
  getInitializer,
  getPadding,
  getRegularizer,
  Initializer,
  Padding,
  Regularizer,
} from '@/app/ir/irCommon';
import { Conv2DOptions } from '@/nodes/model/conv/Conv2D';

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
      options.get(Conv2DOptions.Filters),
      getPadding(options.get(Conv2DOptions.Padding)),
      [getInitializer(options.get(Conv2DOptions.WeightsInitializer)),
        getRegularizer(options.get(Conv2DOptions.WeightsRegularizer))],
      [getInitializer(options.get(Conv2DOptions.BiasInitializer)),
        getRegularizer(options.get(Conv2DOptions.BiasRegularizer))],
      getBuiltinActivationFunction(options.get(Conv2DOptions.Activation)),
      [options.get(Conv2DOptions.KernelSize)[0], options.get(Conv2DOptions.KernelSize)[1]],
      [options.get(Conv2DOptions.Stride)[0], options.get(Conv2DOptions.Stride)[1]],
    );
  }
}
