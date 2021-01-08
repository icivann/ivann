import { MultiMarginLossOptions } from '@/nodes/overview/loss/Multimarginloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class MultiMarginLoss {
  constructor(
  public readonly name: string,
  public readonly P: bigint,
  public readonly Margin: number,
  public readonly Weight: [bigint],
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): MultiMarginLoss {
    return new MultiMarginLoss(

      options.get(nodeName),
      options.get(MultiMarginLossOptions.P),
      options.get(MultiMarginLossOptions.Margin),
      [options.get(MultiMarginLossOptions.Weight)[0]],
      options.get(MultiMarginLossOptions.SizeAverage),
      options.get(MultiMarginLossOptions.Reduce),
      getReduction(options.get(MultiMarginLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `MultiMarginLoss(p=${this.P}, margin=${this.Margin}, weight=${this.Weight}, size_average=${this.SizeAverage}, reduce=${this.Reduce}, reduction=${this.Reduction})`;
  }
}
