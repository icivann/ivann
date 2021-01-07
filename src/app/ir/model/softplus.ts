import { SoftplusOptions } from '@/nodes/model/Softplus';
import { nodeName } from '@/app/ir/irCommon';

export default class Softplus {
  constructor(
  public readonly name: string,
  public readonly Beta: bigint,
  public readonly Threshold: bigint,
  ) {
  }

  static build(options: Map<string, any>): Softplus {
    return new Softplus(

      options.get(nodeName),
      options.get(SoftplusOptions.Beta),
      options.get(SoftplusOptions.Threshold),
    );
  }

  public initCode(): string {
    return `Softplus(Beta=${this.Beta}, Threshold=${this.Threshold})`;
  }
}
