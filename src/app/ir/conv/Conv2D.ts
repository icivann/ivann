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
import { ConvOptions } from '@/nodes/model/conv/Conv';

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
      options.get(ConvOptions.Filters),
      getPadding(options.get(ConvOptions.Padding)),
      [getInitializer(options.get(ConvOptions.WeightsInitializer)),
        getRegularizer(options.get(ConvOptions.WeightsRegularizer))],
      [getInitializer(options.get(ConvOptions.BiasInitializer)),
        getRegularizer(options.get(ConvOptions.BiasRegularizer))],
      getBuiltinActivationFunction(options.get(ConvOptions.Activation)),
      [options.get(ConvOptions.KernelSize)[0], options.get(ConvOptions.KernelSize)[1]],
      [options.get(ConvOptions.Stride)[0], options.get(ConvOptions.Stride)[1]],
    );
  }

  public initCode(): string {
    return `Conv2D(16, ${this.filters}, ${this.stride})`;
  }
}
