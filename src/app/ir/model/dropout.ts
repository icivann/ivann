import { DropoutOptions } from '@/nodes/model/Dropout';
import { nodeName } from '@/app/ir/irCommon';

export default class Dropout {
  constructor(
  public readonly name: string,
  public readonly p: number,
  public readonly inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): Dropout {
    return new Dropout(
      options.get(nodeName),
      options.get(DropoutOptions.P),
      options.get(DropoutOptions.Inplace),
    );
  }

  public initCode(): string {
    return `Dropout(p=${this.p}, inplace=${this.inplace})`;
  }
}
