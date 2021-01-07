import { AdadeltaOptions } from '@/nodes/overview/optimizers/Adadelta';
import { nodeName } from '@/app/ir/irCommon';

export default class Adadelta {
  constructor(
    public readonly name: string,
    public readonly Lr: number,
    public readonly Rho: number,
    public readonly Eps: number,
    public readonly WeightDecay: number,
  ) {
  }

  static build(options: Map<string, any>): Adadelta {
    return new Adadelta(
      options.get(nodeName),
      options.get(AdadeltaOptions.Lr),
      options.get(AdadeltaOptions.Rho),
      options.get(AdadeltaOptions.Eps),
      options.get(AdadeltaOptions.WeightDecay),
    );
  }

  public initCode(params: string): string[] {
    return [`optim.Adadelta(${params[0]}.parameters(), lr=${this.Lr}, rho=${this.Rho}, eps=${this.Eps}, weight_decay=${this.WeightDecay})`];
  }
}
