import { NLLLossOptions } from '@/nodes/overview/loss/Nllloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class NLLLoss {
  constructor(
  public readonly name: string,
  public readonly Weight: [bigint],
  public readonly SizeAverage: bigint,
  public readonly IgnoreIndex: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): NLLLoss {
    return new NLLLoss(

      options.get(nodeName),
      [options.get(NLLLossOptions.Weight)[0]],
      options.get(NLLLossOptions.SizeAverage),
      options.get(NLLLossOptions.IgnoreIndex),
      options.get(NLLLossOptions.Reduce),
      getReduction(options.get(NLLLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `NLLLoss(weight=${this.Weight}, size_average=${this.SizeAverage}, ignore_index=${this.IgnoreIndex}, reduce=${this.Reduce}, reduction=${this.Reduction})`;
  }
}
