import { SoftminOptions } from '@/nodes/model/Softmin';
import { nodeName } from '@/app/ir/irCommon';

export default class Softmin {
  constructor(
  public readonly name: string,
  public readonly dim: [bigint],
  ) {
  }

  static build(options: Map<string, any>): Softmin {
    return new Softmin(
      options.get(nodeName),
      [options.get(SoftminOptions.Dim)[0]],
    );
  }

  public initCode(): string {
    return `Softmin(dim=, ${this.dim})`;
  }
}
