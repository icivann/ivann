import { BCEWithLogitsLossOptions } from '@/nodes/overview/loss/Bcewithlogitsloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class BCEWithLogitsLoss {
  constructor(
  public readonly name: string,
  public readonly Weight: [bigint],
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  public readonly PosWeight: [bigint],
  ) {
  }

  static build(options: Map<string, any>): BCEWithLogitsLoss {
    return new BCEWithLogitsLoss(

      options.get(nodeName),
      [options.get(BCEWithLogitsLossOptions.Weight)[0]],
      options.get(BCEWithLogitsLossOptions.SizeAverage),
      options.get(BCEWithLogitsLossOptions.Reduce),
      getReduction(options.get(BCEWithLogitsLossOptions.Reduction)),
      [options.get(BCEWithLogitsLossOptions.PosWeight)[0]],
    );
  }

  public initCode(): string {
    return `BCEWithLogitsLoss(Weight=${this.Weight}, SizeAverage=${this.SizeAverage}, Reduce=${this.Reduce}, Reduction=${this.Reduction}, PosWeight=${this.PosWeight})`;
  }
}
