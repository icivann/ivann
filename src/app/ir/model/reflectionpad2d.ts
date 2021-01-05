import { ReflectionPad2dOptions } from '@/nodes/model/Reflectionpad2d';
import { nodeName } from '@/app/ir/irCommon';

export default class ReflectionPad2d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint, bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): ReflectionPad2d {
    return new ReflectionPad2d(

      options.get(nodeName),
      [options.get(ReflectionPad2dOptions.Padding)[0], options.get(ReflectionPad2dOptions.Padding)[1],
        options.get(ReflectionPad2dOptions.Padding)[2], options.get(ReflectionPad2dOptions.Padding)[3]],
    );
  }

  public initCode(): string {
    return `ReflectionPad2d(Padding= (${this.Padding}))`;
  }
}
