import { ConstantPad2dOptions } from '@/nodes/model/Constantpad2d';
import { nodeName } from '@/app/ir/irCommon';

export default class ConstantPad2d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint, bigint, bigint],
  public readonly Value: number,
  ) {
  }

  static build(options: Map<string, any>): ConstantPad2d {
    return new ConstantPad2d(

      options.get(nodeName),
      [options.get(ConstantPad2dOptions.Padding)[0], options.get(ConstantPad2dOptions.Padding)[1],
        options.get(ConstantPad2dOptions.Padding)[2], options.get(ConstantPad2dOptions.Padding)[3]],
      options.get(ConstantPad2dOptions.Value),
    );
  }

  public initCode(): string {
    return `ConstantPad2d(padding=(${this.Padding}), value=(${this.Value}))`;
  }
}
