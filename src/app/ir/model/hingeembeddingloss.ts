import { HingeEmbeddingLossOptions } from '@/nodes/overview/loss/Hingeembeddingloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class HingeEmbeddingLoss {
  constructor(
  public readonly name: string,
  public readonly Margin: number,
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): HingeEmbeddingLoss {
    return new HingeEmbeddingLoss(

      options.get(nodeName),
      options.get(HingeEmbeddingLossOptions.Margin),
      options.get(HingeEmbeddingLossOptions.SizeAverage),
      options.get(HingeEmbeddingLossOptions.Reduce),
      getReduction(options.get(HingeEmbeddingLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `HingeEmbeddingLoss(margin=${this.Margin}, size_average=${this.SizeAverage}, reduce=${this.Reduce}, reduction='${this.Reduction}')`;
  }
}
