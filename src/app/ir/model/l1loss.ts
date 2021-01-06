import { L1LossOptions } from '@/nodes/model/L1loss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class L1Loss {
  constructor(
  public readonly name: string,
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): L1Loss {
    return new L1Loss(

      options.get(nodeName),
      options.get(L1LossOptions.SizeAverage),
      options.get(L1LossOptions.Reduce),
      getReduction(options.get(L1LossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `L1Loss(SizeAverage=${this.SizeAverage}, Reduce=${this.Reduce}, Reduction=${this.Reduction})`;
  }
}
