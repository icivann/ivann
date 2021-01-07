import { MaxUnpool3dOptions } from '@/nodes/model/Maxunpool3d';
import { nodeName } from '@/app/ir/irCommon';

export default class MaxUnpool3d {
  constructor(
  public readonly name: string,
  public readonly KernelSize: [bigint, bigint, bigint],
  public readonly Stride: [bigint, bigint, bigint],
  public readonly Padding: [bigint, bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): MaxUnpool3d {
    return new MaxUnpool3d(

      options.get(nodeName),
      [options.get(MaxUnpool3dOptions.KernelSize)[0], options.get(MaxUnpool3dOptions.KernelSize)[1], options.get(MaxUnpool3dOptions.KernelSize)[2]],
      [options.get(MaxUnpool3dOptions.Stride)[0], options.get(MaxUnpool3dOptions.Stride)[1], options.get(MaxUnpool3dOptions.Stride)[2]],
      [options.get(MaxUnpool3dOptions.Padding)[0], options.get(MaxUnpool3dOptions.Padding)[1], options.get(MaxUnpool3dOptions.Padding)[2]],
    );
  }

  public initCode(): string {
    return `MaxUnpool3d(KernelSize=(${this.KernelSize}), Stride=(${this.Stride}), Padding=(${this.Padding}))`;
  }
}
