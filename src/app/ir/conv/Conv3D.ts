import { Option } from '@/app/util';
import {
  ActivationF, Initializer, Padding, Regularizer,
} from '@/app/ir/irCommon';

export default class Conv3D {
  constructor(
    public readonly filters: bigint,
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly activation: ActivationF,

    public readonly kernel: [bigint, bigint, bigint],
    public readonly stride: [bigint, bigint, bigint],
  ) {
  }
  public initCode(): string {
    return `Conv3D(16, ${this.filters}, ${this.stride})`;
  }
}
