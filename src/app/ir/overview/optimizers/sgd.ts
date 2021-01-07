import { SGDOptions } from '@/nodes/overview/optimizers/Sgd';
import { nodeName } from '@/app/ir/irCommon';

export default class SGD {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly Momentum: number,
  public readonly Dampening: number,
  public readonly WeightDecay: number,
  public readonly Nesterov: boolean,
  ) {
  }

  static build(options: Map<string, any>): SGD {
    return new SGD(

      options.get(nodeName),
      options.get(SGDOptions.Lr),
      options.get(SGDOptions.Momentum),
      options.get(SGDOptions.Dampening),
      options.get(SGDOptions.WeightDecay),
      options.get(SGDOptions.Nesterov),
    );
  }

  public initCode(): string {
    return `SGD(lr=${this.Lr}, momentum=${this.Momentum}, dampening=${this.Dampening}, weight_decay=${this.WeightDecay}, nesterov=${this.Nesterov})`;
  }
}
