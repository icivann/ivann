import { UnfoldOptions } from '@/nodes/model/Unfold';
import { nodeName } from '@/app/ir/irCommon';

export default class Unfold {
  constructor(
  public readonly name: string,
  public readonly KernelSize: [bigint],
  public readonly Dilation: [bigint],
  public readonly Padding: [bigint],
  public readonly Stride: [bigint],
  ) {
  }

  static build(options: Map<string, any>): Unfold {
    return new Unfold(

      options.get(nodeName),
      [options.get(UnfoldOptions.KernelSize)[0]],
      [options.get(UnfoldOptions.Dilation)[0]],
      [options.get(UnfoldOptions.Padding)[0]],
      [options.get(UnfoldOptions.Stride)[0]],
    );
  }

  public initCode(): string {
    return `Unfold(kernel_size=${this.KernelSize}, dilation=${this.Dilation}, padding=${this.Padding}, stride=${this.Stride})`;
  }
}
