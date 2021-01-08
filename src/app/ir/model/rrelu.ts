import { RReLUOptions } from '@/nodes/model/Rrelu';
import { nodeName } from '@/app/ir/irCommon';

export default class RReLU {
  constructor(
  public readonly name: string,
  public readonly Lower: number,
  public readonly Upper: number,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): RReLU {
    return new RReLU(

      options.get(nodeName),
      options.get(RReLUOptions.Lower),
      options.get(RReLUOptions.Upper),
      options.get(RReLUOptions.Inplace),
    );
  }

  public initCode(): string {
    return `RReLU(Lower= ${this.Lower}, Upper= ${this.Upper}, Inplace= ${this.Inplace})`;
  }
}
