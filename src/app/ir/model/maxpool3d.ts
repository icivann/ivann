import { MaxPool3dOptions } from '@/nodes/model/Maxpool3d';
import { nodeName } from '@/app/ir/irCommon';

export default class MaxPool3d {
  constructor(
  public readonly name: string,
  public readonly kernel_size: [bigint, bigint, bigint],
  public readonly stride: [bigint, bigint, bigint],
  public readonly padding: [bigint, bigint, bigint],
  public readonly dilation: [bigint, bigint, bigint],
  public readonly return_indices: boolean,
  public readonly ceil_mode: boolean,
  ) {
  }

  static build(options: Map<string, any>): MaxPool3d {
    return new MaxPool3d(
      options.get(nodeName),
      [options.get(MaxPool3dOptions.KernelSize)[0], options.get(MaxPool3dOptions.KernelSize)[1],
        options.get(MaxPool3dOptions.KernelSize)[2]],
      [options.get(MaxPool3dOptions.Stride)[0], options.get(MaxPool3dOptions.Stride)[1],
        options.get(MaxPool3dOptions.Stride)[2]],
      [options.get(MaxPool3dOptions.Padding)[0], options.get(MaxPool3dOptions.Padding)[1],
        options.get(MaxPool3dOptions.Padding)[2]],
      [options.get(MaxPool3dOptions.Dilation)[0], options.get(MaxPool3dOptions.Dilation)[1],
        options.get(MaxPool3dOptions.Dilation)[2]],
      options.get(MaxPool3dOptions.ReturnIndices),
      options.get(MaxPool3dOptions.CeilMode),
    );
  }

  public initCode(): string {
    return `MaxPool3d(kernel_size=(${this.kernel_size}), stride=(${this.stride}), padding=(${this.padding}), dilation=(${this.dilation}), return_indices=${this.return_indices}, ceil_mode=${this.ceil_mode})`;
  }
}
