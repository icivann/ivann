import { PReLUOptions } from '@/nodes/model/Prelu';
import { nodeName } from '@/app/ir/irCommon';

export default class PReLU {
  constructor(
  public readonly name: string,
  public readonly NumParameters: bigint,
  public readonly Init: number,
  ) {
  }

  static build(options: Map<string, any>): PReLU {
    return new PReLU(

      options.get(nodeName),
      options.get(PReLUOptions.NumParameters),
      options.get(PReLUOptions.Init),
    );
  }

  public initCode(): string {
    return `PReLU(NumParameters=${this.NumParameters}, Init=${this.Init})`;
  }
}
