import { AdadeltaOptions } from '@/nodes/overview/optimizers/Adadelta';
import { nodeName } from '@/app/ir/irCommon';

export default class Adadelta {
  constructor(
    public readonly name: string,
    public readonly lr: number,
    public readonly rho: number,
    public readonly eps: number,
    public readonly weight_decay: number,
  ) {
  }

  static build(options: Map<string, any>): Adadelta {
    return new Adadelta(
      options.get(nodeName),
      options.get(AdadeltaOptions.lr),
      options.get(AdadeltaOptions.rho),
      options.get(AdadeltaOptions.eps),
      options.get(AdadeltaOptions.weightDecay),
    );
  }

  public initCode(params: string[]): string[] {
    return [`torch.optim.Adadelta(${params[0]}.parameters(), lr=${this.lr}, rho=${this.rho}, eps=${this.eps}, weight_decay=${this.weight_decay})`];
  }

  public callCode(params: string[], name: string): string[] {
    return [`${name}(${params.join(', ')})`];
  }
}
