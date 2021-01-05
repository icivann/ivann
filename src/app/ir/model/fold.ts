import { FoldOptions } from '@/nodes/model/Fold';
import { nodeName } from '@/app/ir/irCommon';

export default class Fold {
  constructor(
  public readonly name: string,
  public readonly OutputSize: [bigint],
  public readonly KernelSize: [bigint],
  public readonly Dilation: [bigint],
  public readonly Padding: [bigint],
  public readonly Stride: [bigint],
  ) {
  }

  static build(options: Map<string, any>): Fold {
    return new Fold(

      options.get(nodeName),
      [options.get(FoldOptions.OutputSize)[0]],
      [options.get(FoldOptions.KernelSize)[0]],
      [options.get(FoldOptions.Dilation)[0]],
      [options.get(FoldOptions.Padding)[0]],
      [options.get(FoldOptions.Stride)[0]],
    );
  }

  public initCode(): string {
    return `Fold(OutputSize= ${this.OutputSize}, KernelSize= ${this.KernelSize}, Dilation= ${this.Dilation}, Padding= ${this.Padding}, Stride= ${this.Stride})`;
  }
}
