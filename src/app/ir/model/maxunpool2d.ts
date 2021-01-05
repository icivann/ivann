import { MaxUnpool2dOptions } from '@/nodes/model/Maxunpool2d';
import { nodeName } from '@/app/ir/irCommon';

export default class MaxUnpool2d {
  constructor(
  public readonly name: string,
  public readonly KernelSize: [bigint, bigint],
  public readonly Stride: [bigint, bigint],
  public readonly Padding: [bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): MaxUnpool2d {
    return new MaxUnpool2d(

      options.get(nodeName),
      [options.get(MaxUnpool2dOptions.KernelSize)[0], options.get(MaxUnpool2dOptions.KernelSize)[1]],
      [options.get(MaxUnpool2dOptions.Stride)[0], options.get(MaxUnpool2dOptions.Stride)[1]],
      [options.get(MaxUnpool2dOptions.Padding)[0], options.get(MaxUnpool2dOptions.Padding)[1]],
    );
  }

  public initCode(): string {
    return `MaxUnpool2d(KernelSize= (${this.KernelSize}), Stride= (${this.Stride}), Padding= (${this.Padding}))`;
  }
}
