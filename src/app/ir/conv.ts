import { InOutNode, ModelNode } from '@/app/ir/mainNodes';
import { Option, UUID } from '@/app/util';
import {
  ActivationF, Initializer, Padding, Regularizer,
} from '@/app/ir/irCommon';

abstract class Conv implements ModelNode, InOutNode {
  abstract readonly outputs: Set<UUID>;

  abstract readonly input: UUID;

  abstract readonly filters: bigint;

  abstract readonly padding?: Padding;

  abstract readonly weights?: [Initializer, Regularizer]

  abstract readonly biases?: Option<[Initializer, Regularizer]>

  abstract readonly activation: ActivationF
}

export type ConvConstructor = [any, any, any, any, any, any, any, any, any, any]

export class Conv1D extends Conv {
  constructor(
    public readonly filters: bigint,
    public readonly kernel: [bigint],
    public readonly stride: [bigint],
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly dilation: [bigint],
    public readonly activation: ActivationF,
    public readonly outputs: Set<UUID>,
    public readonly input: UUID,
  ) {
    super();
  }
}

export class Conv2D extends Conv {
  constructor(
    public readonly filters: bigint,
    public readonly kernel: [bigint, bigint],
    public readonly stride: [bigint, bigint],
    public readonly activation: ActivationF,
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly input: UUID,
    public readonly outputs: Set<UUID>,
  ) {
    super();
  }
}

export class Conv3D extends Conv {
  constructor(
    public readonly outputs: Set<UUID>,
    public readonly filters: bigint,
    public padding: Padding,
    public weights: [Initializer, Regularizer],
    public readonly biases: Option<[Initializer, Regularizer]>,
    public readonly input: UUID,
    public readonly activation: ActivationF,

    public readonly kernel: [bigint, bigint, bigint],
    public readonly stride: [bigint, bigint, bigint],
  ) {
    super();
  }
}
