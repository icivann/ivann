import { Dropout3dOptions } from '@/nodes/model/Dropout3d';
import { nodeName } from '@/app/ir/irCommon';

export default class Dropout3d {
  constructor(
  public readonly name: string,
  public readonly p: number,
  public readonly inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): Dropout3d {
    return new Dropout3d(
      options.get(nodeName),
      options.get(Dropout3dOptions.P),
      options.get(Dropout3dOptions.Inplace),
    );
  }

  public initCode(): string {
    return `Dropout3d(p=${this.p}, inplace=${this.inplace})`;
  }

  public callCode(params: string[], name: string): string {
    return `${name}(${params.join(', ')})`;
  }
}
