import { MultiLabelSoftMarginLossOptions } from '@/nodes/overview/loss/Multilabelsoftmarginloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class MultiLabelSoftMarginLoss {
  constructor(
  public readonly name: string,
  public readonly Weight: [number],
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
    return `nn.MultiLabelSoftMarginLoss(weight=${this.Weight[0] === 0 ? 'None' : this.Weight}, size_average=${this.SizeAverage}, reduce=${this.Reduce}, reduction='${this.Reduction}')`;
  }
}
