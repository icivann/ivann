import { BCELossOptions } from '@/nodes/overview/loss/Bceloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class BCELoss {
  constructor(
  public readonly name: string,
  public readonly Weight: [number],
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): BCELoss {
    return new BCELoss(

      options.get(nodeName),
      [options.get(BCELossOptions.Weight)[0]],
      options.get(BCELossOptions.SizeAverage),
      options.get(BCELossOptions.Reduce),
      getReduction(options.get(BCELossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `nn.BCELoss(weight=${this.Weight[0] === 0 ? 'None' : this.Weight}, size_average=${this.SizeAverage}, reduce=${this.Reduce}, reduction='${this.Reduction}')`;
  }
}
