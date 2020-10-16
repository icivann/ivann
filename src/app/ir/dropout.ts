import { InOutNode, ModelNode } from '@/app/ir/mainNodes';
import { UUID } from '@/app/util';

export default class Dropout implements ModelNode, InOutNode {
  constructor(
    public readonly input: UUID,
    public readonly outputs: Set<UUID>,

    public readonly probability: number,
  ) {}
  public code(): string {
    const params = [
      `rate=${this.probability}`,
    ];
    return `model.add(layers.Dropout(${params.join(', ')}))`;
  }
}
