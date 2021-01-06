import { MarginRankingLossOptions } from '@/nodes/model/Marginrankingloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class MarginRankingLoss {
  constructor(
  public readonly name: string,
  public readonly Margin: number,
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): MarginRankingLoss {
    return new MarginRankingLoss(

      options.get(nodeName),
      options.get(MarginRankingLossOptions.Margin),
      options.get(MarginRankingLossOptions.SizeAverage),
      options.get(MarginRankingLossOptions.Reduce),
      getReduction(options.get(MarginRankingLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `MarginRankingLoss(Margin=, ${this.Margin}, SizeAverage=, ${this.SizeAverage}, Reduce=, ${this.Reduce}, Reduction=, ${this.Reduction})`;
  }
}
