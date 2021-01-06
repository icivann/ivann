import { ConstantPad1dOptions } from '@/nodes/model/Constantpad1d';
import { nodeName } from '@/app/ir/irCommon';

export default class ConstantPad1d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint],
  public readonly Value: number,
  ) {
  }

  static build(options: Map<string, any>): ConstantPad1d {
    return new ConstantPad1d(

      options.get(nodeName),
      [options.get(ConstantPad1dOptions.Padding)[0],
        options.get(ConstantPad1dOptions.Padding)[1]],
      options.get(ConstantPad1dOptions.Value),
    );
  }

  public initCode(): string {
    return `ConstantPad1d(Padding=, ${this.Padding}, Value=, ${this.Value})`;
  }
}
