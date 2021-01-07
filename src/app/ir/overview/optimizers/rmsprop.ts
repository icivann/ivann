import { RMSpropOptions } from '@/nodes/overview/optimizers/Rmsprop';
import { nodeName } from '@/app/ir/irCommon';

export default class RMSprop {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly Alpha: number,
  public readonly Eps: number,
  public readonly WeightDecay: number,
  public readonly Momentum: number,
  public readonly Centered: boolean,
  ) {
  }

  static build(options: Map<string, any>): RMSprop {
    return new RMSprop(

      options.get(nodeName),
      options.get(RMSpropOptions.Lr),
      options.get(RMSpropOptions.Alpha),
      options.get(RMSpropOptions.Eps),
      options.get(RMSpropOptions.WeightDecay),
      options.get(RMSpropOptions.Momentum),
      options.get(RMSpropOptions.Centered),
    );
  }

  public initCode(params: string[]): string[] {
    return [`optim.RMSprop(${params[0]}.parameters(), lr=${this.Lr}, alpha=${this.Alpha}, eps=${this.Eps}, weight_decay=${this.WeightDecay}, momentum=${this.Momentum}, centered=${this.Centered})`];
  }
}
