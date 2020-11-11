import { Conv2dOptions } from '@/nodes/model/Conv2d';
import { PaddingMode } from '@/app/ir/irCommon';

function getPaddingMode(s: string): PaddingMode {
  return PaddingMode[s as keyof typeof PaddingMode];
}

export default class Conv2d {
  constructor(
    public readonly in_channels: bigint,
    public readonly out_channels: bigint,
    public readonly kernel_size: [bigint, bigint],
    public readonly stride: [bigint, bigint],
    public readonly padding: [bigint, bigint],
    public readonly dilation: [bigint, bigint],
    public readonly groups: bigint,
    public readonly bias: boolean,
    public readonly padding_mode: PaddingMode,
  ) {
  }

  static build(options: Map<string, any>): Conv2d {
    return new Conv2d(
      options.get(Conv2dOptions.InChannels),
      options.get(Conv2dOptions.OutChannels),
      [options.get(Conv2dOptions.KernelSize[0]), options.get(Conv2dOptions.KernelSize)[1]],
      [options.get(Conv2dOptions.Stride[0]), options.get(Conv2dOptions.Stride)[1]],
      [options.get(Conv2dOptions.Padding[0]), options.get(Conv2dOptions.Padding)[1]],
      [options.get(Conv2dOptions.Dilation[0]), options.get(Conv2dOptions.Dilation)[1]],
      options.get(Conv2dOptions.Groups),
      options.get(Conv2dOptions.Bias),
      getPaddingMode(options.get(Conv2dOptions.PaddingMode)),
    );
  }

  public initCode(): string {
    return `Conv2d(in_channels=${this.in_channels}, out_channels=${this.out_channels},
    kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding},
    dilation=${this.dilation}, groups=${this.groups}, bias=${this.bias},
    padding_mode=${this.padding_mode})`;
  }
}
