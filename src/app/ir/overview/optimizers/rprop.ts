import { RpropOptions } from '@/nodes/overview/optimizers/Rprop';
import { nodeName } from '@/app/ir/irCommon';

export default class Rprop {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly Etas: [bigint],
  public readonly StepSizes: [bigint],
  ) {
  }

  static build(options: Map<string, any>): Rprop {
    return new Rprop(

      options.get(nodeName),
      options.get(RpropOptions.Lr),
      [options.get(RpropOptions.Etas)[0]],
      [options.get(RpropOptions.StepSizes)[0]],
    );
  }

  public initCode(params: string[]): string[] {
    return [`optim.Rprop(${params[0]}.parameters(), lr=${this.Lr}, etas=${this.Etas}, step_sizes=${this.StepSizes})`];
  }
}
