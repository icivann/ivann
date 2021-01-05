import { ELUOptions } from '@/nodes/model/Elu';
import { nodeName } from '@/app/ir/irCommon';

export default class ELU {
  constructor(
  public readonly name: string,
  public readonly Alpha: number,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): ELU {
    return new ELU(

      options.get(nodeName),
      options.get(ELUOptions.Alpha),
      options.get(ELUOptions.Inplace),
    );
  }

  public initCode(): string {
    return `ELU(Alpha= ${this.Alpha}, Inplace= ${this.Inplace})`;
  }
}
