import { AdamWOptions } from '@/nodes/overview/optimizers/Adamw';
import { nodeName } from '@/app/ir/irCommon';

export default class AdamW {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly Betas: [bigint],
  public readonly Eps: number,
  public readonly WeightDecay: number,
  public readonly Amsgrad: boolean,
  ) {
  }

  static build(options: Map<string, any>): AdamW {
    return new AdamW(

      options.get(nodeName),
      options.get(AdamWOptions.Lr),
      [options.get(AdamWOptions.Betas)[0]],
      options.get(AdamWOptions.Eps),
      options.get(AdamWOptions.WeightDecay),
      options.get(AdamWOptions.Amsgrad),
    );
  }

  public initCode(params: string): string[] {
    return [`optim.AdamW(${params[0]}.parameters(), lr=${this.Lr}, betas=${this.Betas}, eps=${this.Eps}, weight_decay=${this.WeightDecay}, amsgrad=${this.Amsgrad})`];
  }
}
