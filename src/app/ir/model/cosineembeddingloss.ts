import { CosineEmbeddingLossOptions } from '@/nodes/overview/loss/Cosineembeddingloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class CosineEmbeddingLoss {
  constructor(
  public readonly name: string,
  public readonly Margin: number,
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): CosineEmbeddingLoss {
    return new CosineEmbeddingLoss(

      options.get(nodeName),
      options.get(CosineEmbeddingLossOptions.Margin),
      options.get(CosineEmbeddingLossOptions.SizeAverage),
      options.get(CosineEmbeddingLossOptions.Reduce),
      getReduction(options.get(CosineEmbeddingLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `CosineEmbeddingLoss(margin=${this.Margin}, size_average=${this.SizeAverage}, reduce=${this.Reduce}, reduction=${this.Reduction})`;
  }
}
