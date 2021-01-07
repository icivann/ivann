import { LBFGSOptions } from '@/nodes/overview/optimizers/Lbfgs';
import { nodeName } from '@/app/ir/irCommon';

export default class LBFGS {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly MaxIter: bigint,
  public readonly MaxEval: bigint,
  public readonly ToleranceGrad: number,
  public readonly ToleranceChange: number,
  public readonly HistorySize: bigint,
  ) {
  }

  static build(options: Map<string, any>): LBFGS {
    return new LBFGS(

      options.get(nodeName),
      options.get(LBFGSOptions.Lr),
      options.get(LBFGSOptions.MaxIter),
      options.get(LBFGSOptions.MaxEval),
      options.get(LBFGSOptions.ToleranceGrad),
      options.get(LBFGSOptions.ToleranceChange),
      options.get(LBFGSOptions.HistorySize),
    );
  }

  public initCode(): string {
    return `LBFGS(lr=${this.Lr}, max_iter=${this.MaxIter}, max_eval=${this.MaxEval}, tolerance_grad=${this.ToleranceGrad}, tolerance_change=${this.ToleranceChange}, history_size=${this.HistorySize})`;
  }
}
