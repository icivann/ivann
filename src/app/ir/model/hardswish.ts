import { HardswishOptions } from '@/nodes/model/Hardswish';
import { nodeName } from '@/app/ir/irCommon';

export default class Hardswish {
  constructor(
  public readonly name: string,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): Hardswish {
    return new Hardswish(

      options.get(nodeName),
      options.get(HardswishOptions.Inplace),
    );
  }

  public initCode(): string {
    return `Hardswish(Inplace= ${this.Inplace})`;
  }
}
