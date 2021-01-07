import { LeakyReLUOptions } from '@/nodes/model/Leakyrelu';
import { nodeName } from '@/app/ir/irCommon';

export default class LeakyReLU {
  constructor(
  public readonly name: string,
  public readonly NegativeSlope: number,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): LeakyReLU {
    return new LeakyReLU(

      options.get(nodeName),
      options.get(LeakyReLUOptions.NegativeSlope),
      options.get(LeakyReLUOptions.Inplace),
    );
  }

  public initCode(): string {
    return `LeakyReLU(NegativeSlope=${this.NegativeSlope}, Inplace=${this.Inplace})`;
  }
}
