import { Option } from '@/app/util';
import {
  ActivationF, Initializer, Padding, Regularizer,
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
}
