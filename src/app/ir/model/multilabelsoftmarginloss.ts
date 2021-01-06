import { MultiLabelSoftMarginLossOptions } from '@/nodes/model/Multilabelsoftmarginloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class MultiLabelSoftMarginLoss {
  constructor(
  public readonly name: string,
  public readonly Weight: [bigint],
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): MultiLabelSoftMarginLoss {
    return new MultiLabelSoftMarginLoss(

      options.get(nodeName),
      [options.get(MultiLabelSoftMarginLossOptions.Weight)[0]],
      options.get(MultiLabelSoftMarginLossOptions.SizeAverage),
      options.get(MultiLabelSoftMarginLossOptions.Reduce),
      getReduction(options.get(MultiLabelSoftMarginLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `MultiLabelSoftMarginLoss(Weight=, ${this.Weight}, SizeAverage=, ${this.SizeAverage}, Reduce=, ${this.Reduce}, Reduction=, ${this.Reduction})`;
  }
}
