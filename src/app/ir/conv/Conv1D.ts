import { Option } from '@/app/util';
import {
  ActivationF, BuiltinActivationF, Initializer, Padding, Regularizer,
} from '@/app/ir/irCommon';

export default class Conv1D {
  constructor(
    public readonly filters: bigint,
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly dilation: [bigint],
    public readonly activation: ActivationF,
    public readonly kernel: [bigint],
    public readonly stride: [bigint],
  ) {}

  static build(options: Map<string, any>): Conv1D {
    console.log(options);
    const node = new Conv1D(
      options.get('Filters'),
      options.get('Padding'),
      [options.get('Weights Initializer'), options.get('Weights Regularizer')],
      [options.get('Bias Initializer'), options.get('Bias Regularizer')],
      options.get('Dilatation'),
      BuiltinActivationF[options.get('Activation') as keyof typeof BuiltinActivationF],
      [options.get('Kernel Size')[0]],
      [options.get('Stride')[0]],
    );
    return node;
  }

  public initCode(): string {
    return `Conv1D(16, ${this.filters}, ${this.stride})`;
  }

  public callCode(params: string[], name: string): string {
    return `self.${name}(${params.join(', ')})`;
  }
}
