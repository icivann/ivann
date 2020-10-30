import { Conv2dOptions } from '@/nodes/pytorch model/Conv2dBaklava';

enum Padding_mode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPadding_mode(s: string) : Padding_mode {
  return Padding_mode[s as keyof typeof Padding_mode];
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
  public readonly padding_mode: Padding_mode,
) {
}

static build(options: Map<string, any>): Conv2d {
  return new Conv2d(
    options.get(Conv2dOptions.In_channels),
  options.get(Conv2dOptions.Out_channels),
  [  options.get(Conv2dOptions.Kernel_size[0]), options.get(Conv2dOptions.Kernel_size)[1]],
  [  options.get(Conv2dOptions.Stride[0]), options.get(Conv2dOptions.Stride)[1]],
  [  options.get(Conv2dOptions.Padding[0]), options.get(Conv2dOptions.Padding)[1]],
  [  options.get(Conv2dOptions.Dilation[0]), options.get(Conv2dOptions.Dilation)[1]],
  options.get(Conv2dOptions.Groups),
  options.get(Conv2dOptions.Bias),
  getPadding_mode(options.get(Conv2dOptions.Padding_mode))
  );

  }

  public initCode(): string{
    return `Conv2d(in_channels=${this.in_channels}, out_channels=${this.out_channels}, kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, dilation=${this.dilation}, groups=${this.groups}, bias=${this.bias}, padding_mode=${this.padding_mode})`;
  }
}
