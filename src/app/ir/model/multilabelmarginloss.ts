import { MultiLabelMarginLossOptions } from '@/nodes/model/Multilabelmarginloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class MultiLabelMarginLoss {
  constructor(
  public readonly name: string,
  public readonly SizeAverage: bigint,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): MultiLabelMarginLoss {
    return new MultiLabelMarginLoss(

      options.get(nodeName),
      options.get(MultiLabelMarginLossOptions.SizeAverage),
      options.get(MultiLabelMarginLossOptions.Reduce),
      getReduction(options.get(MultiLabelMarginLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `MultiLabelMarginLoss(SizeAverage= ${this.SizeAverage}, Reduce= ${this.Reduce}, Reduction= ${this.Reduction})`;
  }
}
