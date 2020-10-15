import { InOutNode, ModelNode } from '@/app/ir/mainNodes';
import { Option, UUID } from '@/app/util';
import {
  ActivationF, Initializer, Padding, Regularizer,
} from '@/app/ir/irCommon';

export default abstract class Conv implements ModelNode, InOutNode {
  abstract readonly outputs: Set<UUID>;

  abstract readonly input: UUID;

  abstract readonly filters: bigint;

  abstract readonly padding: Padding;

  abstract readonly weights: [Initializer, Regularizer]

  abstract readonly biases: Option<[Initializer, Regularizer]>

  abstract readonly activation: ActivationF

  public abstract code(): string;
}
