import { KLDivLossOptions } from '@/nodes/model/Kldivloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class KLDivLoss {
  constructor(
  public readonly name: string,
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  public readonly LogTarget: boolean,
  ) {
  }

  static build(options: Map<string, any>): KLDivLoss {
    return new KLDivLoss(

      options.get(nodeName),
      options.get(KLDivLossOptions.SizeAverage),
      options.get(KLDivLossOptions.Reduce),
      getReduction(options.get(KLDivLossOptions.Reduction)),
      options.get(KLDivLossOptions.LogTarget),
    );
  }

  public initCode(): string {
    return `KLDivLoss(SizeAverage= ${this.SizeAverage}, Reduce= ${this.Reduce}, Reduction= ${this.Reduction}, LogTarget= ${this.LogTarget})`;
  }
}
