import { Option, UUID } from '@/app/util';
import {
  ActivationF, Initializer, Padding, Regularizer,
} from '@/app/ir/irCommon';
import Conv from '@/app/ir/Conv';

export default class Conv1D extends Conv {
  constructor(
    public readonly outputs: Set<UUID>,
    public readonly filters: bigint,
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly input: UUID,
    public readonly dilation: [bigint],
    public readonly activation: ActivationF,

    public readonly kernel: [bigint],
    public readonly stride: [bigint],
  ) {
    super();
  }
}
