import { InOutNode, ModelNode } from '@/app/ir/mainNodes';
import { Option, UUID } from '@/app/util';
import {
  ActivationF, Initializer, Padding, Regularizer,
} from '@/app/ir/irCommon';

abstract class Conv implements ModelNode, InOutNode {
  abstract readonly out: UUID

  abstract readonly input: UUID

  abstract readonly filters: bigint

  abstract readonly padding: Padding

  abstract readonly weights: [Initializer, Regularizer]

  abstract readonly biases: Option<[Initializer, Regularizer]>

  abstract readonly activation: ActivationF
}

class Conv1D extends Conv {
  constructor(
    public readonly out: UUID,
    public readonly filters: bigint,
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly input: UUID,
    public readonly dilation: [bigint, bigint],
    public readonly activation: ActivationF,
  ) {
    super();
  }
}
