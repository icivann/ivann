import { TripletMarginLossOptions } from '@/nodes/overview/loss/Tripletmarginloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class TripletMarginLoss {
  constructor(
  public readonly name: string,
  public readonly Margin: number,
  public readonly P: number,
  public readonly Eps: number,
  public readonly Swap: boolean,
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): TripletMarginLoss {
    return new TripletMarginLoss(

      options.get(nodeName),
      options.get(TripletMarginLossOptions.Margin),
      options.get(TripletMarginLossOptions.P),
      options.get(TripletMarginLossOptions.Eps),
      options.get(TripletMarginLossOptions.Swap),
      options.get(TripletMarginLossOptions.SizeAverage),
      options.get(TripletMarginLossOptions.Reduce),
      getReduction(options.get(TripletMarginLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `TripletMarginLoss(Margin=${this.Margin}, P=${this.P}, Eps=${this.Eps}, Swap=${this.Swap}, SizeAverage=${this.SizeAverage}, Reduce=${this.Reduce}, Reduction=${this.Reduction})`;
  }
}
