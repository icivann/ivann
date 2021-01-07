import { SparseAdamOptions } from '@/nodes/overview/optimizers/Sparseadam';
import { nodeName } from '@/app/ir/irCommon';

export default class SparseAdam {
  constructor(
  public readonly name: string,
  public readonly Lr: number,
  public readonly Betas: [bigint],
  public readonly Eps: number,
  ) {
  }

  static build(options: Map<string, any>): SparseAdam {
    return new SparseAdam(

      options.get(nodeName),
      options.get(SparseAdamOptions.Lr),
      [options.get(SparseAdamOptions.Betas)[0]],
      options.get(SparseAdamOptions.Eps),
    );
  }

  public initCode(params: string): string[] {
    return [`optim.SparseAdam(${params[0]}.parameters(), lr=${this.Lr}, betas=${this.Betas}, eps=${this.Eps})`];
  }
}
