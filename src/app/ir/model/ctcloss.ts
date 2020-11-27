import { CTCLossOptions } from '@/nodes/model/Ctcloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class CTCLoss {
  constructor(
  public readonly name: string,
  public readonly Blank: bigint,
  public readonly Reduction: Reduction,
  public readonly ZeroInfinity: boolean,
  ) {
  }

  static build(options: Map<string, any>): CTCLoss {
    return new CTCLoss(

      options.get(nodeName),
      options.get(CTCLossOptions.Blank),
      getReduction(options.get(CTCLossOptions.Reduction)),
      options.get(CTCLossOptions.ZeroInfinity),
    );
  }

  public initCode(): string {
    return `CTCLoss(Blank=, ${this.Blank}, Reduction=, ${this.Reduction}, ZeroInfinity=, ${this.ZeroInfinity})`;
  }
}
