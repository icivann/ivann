import { MaxUnpool1dOptions } from '@/nodes/model/Maxunpool1d';
import { nodeName } from '@/app/ir/irCommon';

export default class MaxUnpool1d {
  constructor(
  public readonly name: string,
  public readonly KernelSize: [bigint],
  public readonly Stride: [bigint],
  public readonly Padding: [bigint],
  ) {
  }

  static build(options: Map<string, any>): MaxUnpool1d {
    return new MaxUnpool1d(

      options.get(nodeName),
      [options.get(MaxUnpool1dOptions.KernelSize)[0]],
      [options.get(MaxUnpool1dOptions.Stride)[0]],
      [options.get(MaxUnpool1dOptions.Padding)[0]],
    );
  }

  public initCode(): string {
    return `MaxUnpool1d(KernelSize=, ${this.KernelSize}, Stride=, ${this.Stride}, Padding=, ${this.Padding})`;
  }
}
