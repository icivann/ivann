import { CELUOptions } from '@/nodes/model/Celu';
import { nodeName } from '@/app/ir/irCommon';

export default class CELU {
  constructor(
  public readonly name: string,
  public readonly Alpha: number,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): CELU {
    return new CELU(

      options.get(nodeName),
      options.get(CELUOptions.Alpha),
      options.get(CELUOptions.Inplace),
    );
  }

  public initCode(): string {
    return `CELU(alpha=${this.Alpha}, inplace=${this.Inplace})`;
  }
}
