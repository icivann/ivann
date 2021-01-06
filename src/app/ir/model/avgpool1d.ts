import { AvgPool1dOptions } from '@/nodes/model/Avgpool1d';
import { nodeName } from '@/app/ir/irCommon';

export default class AvgPool1d {
  constructor(
  public readonly name: string,
  public readonly KernelSize: [bigint],
  public readonly Stride: [bigint],
  public readonly Padding: [bigint],
  public readonly CeilMode: boolean,
  public readonly CountIncludePad: boolean,
  ) {
  }

  static build(options: Map<string, any>): AvgPool1d {
    return new AvgPool1d(

      options.get(nodeName),
      [options.get(AvgPool1dOptions.KernelSize)[0]],
      [options.get(AvgPool1dOptions.Stride)[0]],
      [options.get(AvgPool1dOptions.Padding)[0]],
      options.get(AvgPool1dOptions.CeilMode),
      options.get(AvgPool1dOptions.CountIncludePad),
    );
  }

  public initCode(): string {
    return `AvgPool1d(KernelSize=, ${this.KernelSize}, Stride=, ${this.Stride}, Padding=, ${this.Padding}, CeilMode=, ${this.CeilMode}, CountIncludePad=, ${this.CountIncludePad})`;
  }
}
