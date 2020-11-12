import { Dropout2dOptions } from '@/nodes/model/Dropout2d';
import { nodeName } from '@/app/ir/irCommon';

export default class Dropout2d {
  constructor(
  public readonly name: string,
  public readonly p: number,
  public readonly inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): Dropout2d {
    return new Dropout2d(
      options.get(nodeName),
      options.get(Dropout2dOptions.P),
      options.get(Dropout2dOptions.Inplace),
    );
  }

  public initCode(): string {
    return `Dropout2d(p=${this.p}, inplace=${this.inplace})`;
  }
}
