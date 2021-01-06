import { ConstantPad3dOptions } from '@/nodes/model/Constantpad3d';
import { nodeName } from '@/app/ir/irCommon';

export default class ConstantPad3d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint, bigint, bigint, bigint, bigint],
  public readonly Value: number,
  ) {
  }

  static build(options: Map<string, any>): ConstantPad3d {
    return new ConstantPad3d(

      options.get(nodeName),
      [options.get(ConstantPad3dOptions.Padding)[0], options.get(ConstantPad3dOptions.Padding)[1], options.get(ConstantPad3dOptions.Padding)[2],
        options.get(ConstantPad3dOptions.Padding)[3], options.get(ConstantPad3dOptions.Padding)[4], options.get(ConstantPad3dOptions.Padding)[5]],
      options.get(ConstantPad3dOptions.Value),
    );
  }

  public initCode(): string {
    return `ConstantPad3d(Padding=(${this.Padding}), Value=(${this.Value}))`;
  }
}
