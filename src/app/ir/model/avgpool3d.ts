import { AvgPool3dOptions } from '@/nodes/model/Avgpool3d';
import { nodeName } from '@/app/ir/irCommon';

export default class AvgPool3d {
  constructor(
  public readonly name: string,
  public readonly KernelSize: [bigint, bigint, bigint],
  public readonly Stride: [bigint, bigint, bigint],
  public readonly Padding: [bigint, bigint, bigint],
  public readonly CeilMode: boolean,
  public readonly CountIncludePad: boolean,
  ) {
  }

  static build(options: Map<string, any>): AvgPool3d {
    return new AvgPool3d(

      options.get(nodeName),
      [options.get(AvgPool3dOptions.KernelSize)[0], options.get(AvgPool3dOptions.KernelSize)[1], options.get(AvgPool3dOptions.KernelSize)[2]],
      [options.get(AvgPool3dOptions.Stride)[0], options.get(AvgPool3dOptions.Stride)[1], options.get(AvgPool3dOptions.Stride)[2]],
      [options.get(AvgPool3dOptions.Padding)[0], options.get(AvgPool3dOptions.Padding)[1], options.get(AvgPool3dOptions.Padding)[2]],
      options.get(AvgPool3dOptions.CeilMode),
      options.get(AvgPool3dOptions.CountIncludePad),
    );
  }

  public initCode(): string {
    return `AvgPool3d(KernelSize= (${this.KernelSize}), Stride= (${this.Stride}), Padding= (${this.Padding}), CeilMode= (${this.CeilMode}), CountIncludePad= (${this.CountIncludePad}))`;
  }
}
