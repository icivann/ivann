import { MSELossOptions } from '@/nodes/overview/loss/Mseloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class MSELoss {
  constructor(
  public readonly name: string,
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): MSELoss {
    return new MSELoss(

      options.get(nodeName),
      options.get(MSELossOptions.SizeAverage),
      options.get(MSELossOptions.Reduce),
      getReduction(options.get(MSELossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `MSELoss(size_average=${this.SizeAverage}, reduce=${this.Reduce}, reduction=${this.Reduction})`;
  }
}
