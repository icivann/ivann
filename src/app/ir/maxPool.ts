import { InOutNode, ModelNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';
import { Padding } from '@/app/ir/irCommon';

abstract class MaxPool implements ModelNode, InOutNode {
  abstract readonly out: UUID;

  abstract readonly input: UUID;
}

class MaxPool1D extends MaxPool {
  constructor(
    public readonly out: UUID,
    public readonly input: UUID,
    public padding: Padding,
    public readonly kernel: [bigint],
    public readonly stride: [bigint],
  ) {
    super();
  }
}

class MaxPool2D extends MaxPool {
  constructor(
    public readonly out: UUID,
    public readonly input: UUID,
    public padding: Padding,
    public readonly kernel: [bigint, bigint],
    public readonly stride: [bigint, bigint],
  ) {
    super();
  }
}

class MaxPool3D extends MaxPool {
  constructor(
    public readonly out: UUID,
    public readonly input: UUID,
    public padding: Padding,
    public readonly kernel: [bigint, bigint, bigint],
    public readonly stride: [bigint, bigint, bigint],
  ) {
    super();
  }
}
