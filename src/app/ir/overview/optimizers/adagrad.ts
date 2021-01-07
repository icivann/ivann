import { AdagradOptions } from '@/nodes/overview/optimizers/Adagrad';
import { nodeName } from '@/app/ir/irCommon';

export default class Adagrad {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly LrDecay: number,
  public readonly WeightDecay: number,
  public readonly InitialAccumulatorValue: number,
  public readonly Eps: number,
  ) {
  }

  static build(options: Map<string, any>): Adagrad {
    return new Adagrad(

      options.get(nodeName),
      options.get(AdagradOptions.Lr),
      options.get(AdagradOptions.LrDecay),
      options.get(AdagradOptions.WeightDecay),
      options.get(AdagradOptions.InitialAccumulatorValue),
      options.get(AdagradOptions.Eps),
    );
  }

  public initCode(params: string): string[] {
    return [`optim.Adagrad(${params[0]}.parameters(), lr=${this.Lr}, lr_decay=${this.LrDecay}, weight_decay=${this.WeightDecay}, initial_accumulator_value=${this.InitialAccumulatorValue}, eps=${this.Eps})`];
  }
}
