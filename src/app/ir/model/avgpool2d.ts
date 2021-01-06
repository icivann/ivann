import { AvgPool2dOptions } from '@/nodes/model/Avgpool2d';
import { nodeName } from '@/app/ir/irCommon';

export default class AvgPool2d {
  constructor(
  public readonly name: string,
  public readonly KernelSize: [bigint, bigint],
  public readonly Stride: [bigint, bigint],
  public readonly Padding: [bigint, bigint],
  public readonly CeilMode: boolean,
  public readonly CountIncludePad: boolean,
  public readonly DivisorOverride: boolean,
  ) {
  }

  static build(options: Map<string, any>): AvgPool2d {
    return new AvgPool2d(

      options.get(nodeName),
      [options.get(AvgPool2dOptions.KernelSize)[0], options.get(AvgPool2dOptions.KernelSize)[1]],
      [options.get(AvgPool2dOptions.Stride)[0], options.get(AvgPool2dOptions.Stride)[1]],
      [options.get(AvgPool2dOptions.Padding)[0], options.get(AvgPool2dOptions.Padding)[1]],
      options.get(AvgPool2dOptions.CeilMode),
      options.get(AvgPool2dOptions.CountIncludePad),
      options.get(AvgPool2dOptions.DivisorOverride),
    );
  }

  public initCode(): string {
    return `AvgPool2d(KernelSize=(${this.KernelSize}), Stride=(${this.Stride}), Padding=(${this.Padding}), CeilMode=(${this.CeilMode}), CountIncludePad=(${this.CountIncludePad}), DivisorOverride=(${this.DivisorOverride}))`;
  }
}
