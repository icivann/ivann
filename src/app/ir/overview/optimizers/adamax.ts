import { AdamaxOptions } from '@/nodes/overview/optimizers/Adamax';
import { nodeName } from '@/app/ir/irCommon';

export default class Adamax {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly Betas: [bigint],
  public readonly Eps: number,
  public readonly WeightDecay: number,
  ) {
  }

  static build(options: Map<string, any>): Adamax {
    return new Adamax(

      options.get(nodeName),
      options.get(AdamaxOptions.Lr),
      [options.get(AdamaxOptions.Betas)[0]],
      options.get(AdamaxOptions.Eps),
      options.get(AdamaxOptions.WeightDecay),
    );
  }

  public initCode(params: string): string[] {
    return [`optim.Adamax(${params[0]}.parameters(), lr=${this.Lr}, betas=${this.Betas}, eps=${this.Eps}, weight_decay=${this.WeightDecay})`];
  }
}
