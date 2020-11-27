import { ZeroPad2dOptions } from '@/nodes/model/Zeropad2d';
import { nodeName } from '@/app/ir/irCommon';

export default class ZeroPad2d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint, bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): ZeroPad2d {
    return new ZeroPad2d(

      options.get(nodeName),
      [options.get(ZeroPad2dOptions.Padding)[0], options.get(ZeroPad2dOptions.Padding)[1],
        options.get(ZeroPad2dOptions.Padding)[2], options.get(ZeroPad2dOptions.Padding)[3]],
    );
  }

  public initCode(): string {
    return `ZeroPad2d(Padding=, (${this.Padding}))`;
  }
}
