import { LPPool2dOptions } from '@/nodes/model/Lppool2d';
import { nodeName } from '@/app/ir/irCommon';

export default class LPPool2d {
  constructor(
  public readonly name: string,
  public readonly NormType: number,
  public readonly KernelSize: [bigint, bigint],
  public readonly Stride: [bigint, bigint],
  public readonly CeilMode: boolean,
  ) {
  }

  static build(options: Map<string, any>): LPPool2d {
    return new LPPool2d(

      options.get(nodeName),
      options.get(LPPool2dOptions.NormType),
      [options.get(LPPool2dOptions.KernelSize)[0], options.get(LPPool2dOptions.KernelSize)[1]],
      [options.get(LPPool2dOptions.Stride)[0], options.get(LPPool2dOptions.Stride)[1]],
      options.get(LPPool2dOptions.CeilMode),
    );
  }

  public initCode(): string {
    return `LPPool2d(NormType=(${this.NormType}), KernelSize=(${this.KernelSize}), Stride=(${this.Stride}), CeilMode=(${this.CeilMode}))`;
  }
}
