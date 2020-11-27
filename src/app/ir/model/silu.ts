import { SiLUOptions } from '@/nodes/model/Silu';
import { nodeName } from '@/app/ir/irCommon';

export default class SiLU {
  constructor(
  public readonly name: string,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): SiLU {
    return new SiLU(

      options.get(nodeName),
      options.get(SiLUOptions.Inplace),
    );
  }

  public initCode(): string {
    return `SiLU(Inplace=, ${this.Inplace})`;
  }
}
