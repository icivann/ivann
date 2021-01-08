import { CrossEntropyLossOptions } from '@/nodes/overview/loss/Crossentropyloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class CrossEntropyLoss {
  constructor(
  public readonly name: string,
  public readonly Weight: [bigint],
  public readonly SizeAverage: bigint,
  public readonly IgnoreIndex: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): CrossEntropyLoss {
    return new CrossEntropyLoss(

      options.get(nodeName),
      [options.get(CrossEntropyLossOptions.Weight)[0]],
      options.get(CrossEntropyLossOptions.SizeAverage),
      options.get(CrossEntropyLossOptions.IgnoreIndex),
      options.get(CrossEntropyLossOptions.Reduce),
      getReduction(options.get(CrossEntropyLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `CrossEntropyLoss(weight=${this.Weight}, size_average=${this.SizeAverage}, ignore_index=${this.IgnoreIndex}, reduce=${this.Reduce}, reduction='${this.Reduction}')`;
  }
}
