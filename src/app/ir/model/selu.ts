import { SELUOptions } from '@/nodes/model/Selu';
import { nodeName } from '@/app/ir/irCommon';

export default class SELU {
  constructor(
  public readonly name: string,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): SELU {
    return new SELU(

      options.get(nodeName),
      options.get(SELUOptions.Inplace),
    );
  }

  public initCode(): string {
    return `SELU(inplace=${this.Inplace})`;
  }
}
