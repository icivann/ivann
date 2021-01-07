import { AdamOptions } from '@/nodes/overview/optimizers/Adam';
import { nodeName } from '@/app/ir/irCommon';

export default class Adam {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly Betas: [bigint],
  public readonly Eps: number,
  public readonly WeightDecay: number,
  public readonly Amsgrad: boolean,
  ) {
  }

  static build(options: Map<string, any>): Adam {
    return new Adam(

      options.get(nodeName),
      options.get(AdamOptions.Lr),
      [options.get(AdamOptions.Betas)[0]],
      options.get(AdamOptions.Eps),
      options.get(AdamOptions.WeightDecay),
      options.get(AdamOptions.Amsgrad),
    );
  }

  public initCode(params: string): string[] {
    return [`optim.Adam(${params[0]}.parameters(), lr=${this.Lr}, betas=${this.Betas}, eps=${this.Eps}, weight_decay=${this.WeightDecay}, amsgrad=${this.Amsgrad})`];
  }
}
