import { PoissonNLLLossOptions } from '@/nodes/overview/loss/Poissonnllloss';
import { nodeName, Reduction, getReduction } from '@/app/ir/irCommon';

export default class PoissonNLLLoss {
  constructor(
  public readonly name: string,
  public readonly LogInput: boolean,
  public readonly Full: boolean,
  public readonly SizeAverage: bigint,
  public readonly Eps: number,
  public readonly Reduce: bigint,
  public readonly Reduction: Reduction,
  ) {
  }

  static build(options: Map<string, any>): PoissonNLLLoss {
    return new PoissonNLLLoss(

      options.get(nodeName),
      options.get(PoissonNLLLossOptions.LogInput),
      options.get(PoissonNLLLossOptions.Full),
      options.get(PoissonNLLLossOptions.SizeAverage),
      options.get(PoissonNLLLossOptions.Eps),
      options.get(PoissonNLLLossOptions.Reduce),
      getReduction(options.get(PoissonNLLLossOptions.Reduction)),
    );
  }

  public initCode(): string {
    return `PoissonNLLLoss(log_input=${this.LogInput}, full=${this.Full}, size_average=${this.SizeAverage}, eps=${this.Eps}, reduce=${this.Reduce}, reduction=${this.Reduction})`;
  }
}
