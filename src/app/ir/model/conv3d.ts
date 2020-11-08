import { Conv3dOptions } from '@/nodes/model/Conv3dBaklava';

enum PaddingMode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPaddingMode(s: string): PaddingMode {
  return PaddingMode[s as keyof typeof PaddingMode];
}

export default class Conv3d {
  constructor(
  public readonly in_channels: bigint,
  public readonly out_channels: bigint,
  public readonly kernel_size: [bigint, bigint, bigint],
  public readonly stride: [bigint, bigint, bigint],
  public readonly padding: [bigint, bigint, bigint],
  public readonly dilation: [bigint, bigint, bigint],
  public readonly groups: bigint,
  public readonly bias: boolean,
  public readonly padding_mode: PaddingMode,
  ) {
  }

  static build(options: Map<string, any>): Conv3d {
    return new Conv3d(
      options.get(Conv3dOptions.InChannels),
      options.get(Conv3dOptions.OutChannels),
      [options.get(Conv3dOptions.KernelSize[0]), options.get(Conv3dOptions.KernelSize[1]),
        options.get(Conv3dOptions.KernelSize)[2]], [options.get(Conv3dOptions.Stride[0]),
        options.get(Conv3dOptions.Stride[1]), options.get(Conv3dOptions.Stride)[2]],
      [options.get(Conv3dOptions.Padding[0]), options.get(Conv3dOptions.Padding[1]),
        options.get(Conv3dOptions.Padding)[2]], [options.get(Conv3dOptions.Dilation[0]),
        options.get(Conv3dOptions.Dilation[1]), options.get(Conv3dOptions.Dilation)[2]],
      options.get(Conv3dOptions.Groups),
      options.get(Conv3dOptions.Bias),
      getPaddingMode(options.get(Conv3dOptions.PaddingMode)),
    );
  }

  public initCode(): string {
    return `Conv3d(in_channels=${this.in_channels}, out_channels=${this.out_channels},
    kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding},
    dilation=${this.dilation}, groups=${this.groups}, bias=${this.bias},
  padding_mode=${this.padding_mode})`;
  }

  public callCode(params: string[], name: string): string {
    return `${name}(${params.join(', ')})`;
  }
}
