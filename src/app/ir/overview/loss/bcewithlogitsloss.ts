import { BCEWithLogitsLossOptions } from '@/nodes/overview/loss/Bcewithlogitsloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class BCEWithLogitsLoss {
  constructor(
  public readonly name: string,
  public readonly Weight: [number],
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  public readonly PosWeight: [number],
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
    return `nn.BCEWithLogitsLoss(weight=${this.Weight[0] === 0 ? 'None' : this.Weight}, size_average=${this.SizeAverage}, reduce=${this.Reduce}, reduction='${this.Reduction}', pos_weight=${this.PosWeight[0] === 0 ? 'None' : this.PosWeight})`;
  }
}
