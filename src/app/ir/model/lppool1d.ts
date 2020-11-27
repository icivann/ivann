import { LPPool1dOptions } from '@/nodes/model/Lppool1d';
import { nodeName } from '@/app/ir/irCommon';

export default class LPPool1d {
  constructor(
  public readonly name: string,
  public readonly NormType: number,
  public readonly KernelSize: [bigint],
  public readonly Stride: [bigint],
  public readonly CeilMode: boolean,
  ) {
  }

  static build(options: Map<string, any>): LPPool1d {
    return new LPPool1d(

      options.get(nodeName),
      options.get(LPPool1dOptions.NormType),
      [options.get(LPPool1dOptions.KernelSize)[0]],
      [options.get(LPPool1dOptions.Stride)[0]],
      options.get(LPPool1dOptions.CeilMode),
    );
  }

  public initCode(): string {
    return `LPPool1d(NormType=, ${this.NormType}, KernelSize=, ${this.KernelSize}, Stride=, ${this.Stride}, CeilMode=, ${this.CeilMode})`;
  }
}
