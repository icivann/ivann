import { CrossEntropyLossOptions } from '@/nodes/model/Crossentropyloss';
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
    return `CrossEntropyLoss(Weight=, ${this.Weight}, SizeAverage=, ${this.SizeAverage}, IgnoreIndex=, ${this.IgnoreIndex}, Reduce=, ${this.Reduce}, Reduction=, ${this.Reduction})`;
  }
}
