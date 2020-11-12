import { MaxPool1dOptions } from '@/nodes/model/Maxpool1d';
import { Conv1dOptions } from '@/nodes/model/Conv1d';
import { nodeName } from '@/app/ir/irCommon';

export default class MaxPool1d {
  constructor(
  public readonly name: string,
  public readonly kernel_size: [bigint],
  public readonly stride: [bigint],
  public readonly padding: [bigint],
  public readonly dilation: [bigint],
  public readonly return_indices: boolean,
  public readonly ceil_mode: boolean,
  ) {
  }

  static build(options: Map<string, any>): MaxPool1d {
    return new MaxPool1d(
      options.get(nodeName),
      [options.get(MaxPool1dOptions.KernelSize)[0]],
      [options.get(MaxPool1dOptions.Stride)[0]],
      [options.get(MaxPool1dOptions.Padding)[0]],
      [options.get(MaxPool1dOptions.Dilation)[0]],
      options.get(MaxPool1dOptions.ReturnIndices),
      options.get(MaxPool1dOptions.CeilMode),
    );
  }

  public initCode(): string {
    return `MaxPool1d(kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, dilation=${this.dilation}, return_indices=${this.return_indices}, ceil_mode=${this.ceil_mode})`;
  }
}
