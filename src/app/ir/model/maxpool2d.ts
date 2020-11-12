import { MaxPool2dOptions } from '@/nodes/model/Maxpool2d';
import { nodeName } from '@/app/ir/irCommon';

export default class MaxPool2d {
  constructor(
  public readonly name: string,
  public readonly kernel_size: [bigint, bigint],
  public readonly stride: [bigint, bigint],
  public readonly padding: [bigint, bigint],
  public readonly dilation: [bigint, bigint],
  public readonly return_indices: boolean,
  public readonly ceil_mode: boolean,
  ) {
  }

  static build(options: Map<string, any>): MaxPool2d {
    return new MaxPool2d(
      options.get(nodeName),
      [options.get(MaxPool2dOptions.KernelSize)[0], options.get(MaxPool2dOptions.KernelSize)[1]],
      [options.get(MaxPool2dOptions.Stride)[0], options.get(MaxPool2dOptions.Stride)[1]],
      [options.get(MaxPool2dOptions.Padding)[0], options.get(MaxPool2dOptions.Padding)[1]],
      [options.get(MaxPool2dOptions.Dilation)[0], options.get(MaxPool2dOptions.Dilation)[1]],
      options.get(MaxPool2dOptions.ReturnIndices),
      options.get(MaxPool2dOptions.CeilMode),
    );
  }

  public initCode(): string {
    return `MaxPool2d(kernel_size=(${this.kernel_size}), stride=(${this.stride}), padding=(${this.padding}), dilation=(${this.dilation}), return_indices=${this.return_indices}, ceil_mode=${this.ceil_mode})`;
  }
}
