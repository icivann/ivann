import { ASGDOptions } from '@/nodes/overview/optimizers/Asgd';
import { nodeName } from '@/app/ir/irCommon';

export default class ASGD {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly Lambd: number,
  public readonly Alpha: number,
  public readonly T0: number,
  public readonly WeightDecay: number,
  ) {
  }

  static build(options: Map<string, any>): ASGD {
    return new ASGD(

      options.get(nodeName),
      options.get(ASGDOptions.Lr),
      options.get(ASGDOptions.Lambd),
      options.get(ASGDOptions.Alpha),
      options.get(ASGDOptions.T0),
      options.get(ASGDOptions.WeightDecay),
    );
  }

  public initCode(): string {
    return `ASGD(lr=${this.Lr}, lambd=${this.Lambd}, alpha=${this.Alpha}, t0=${this.T0}, weight_decay=${this.WeightDecay})`;
  }
}
