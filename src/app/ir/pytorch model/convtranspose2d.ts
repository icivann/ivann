import { ConvTranspose2dOptions } from '@/nodes/pytorch model/ConvTranspose2dBaklava';

enum Padding_mode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPadding_mode(s: string) : Padding_mode {
  return Padding_mode[s as keyof typeof Padding_mode];
}
export default class ConvTranspose2d {
constructor(
  public readonly in_channels: bigint,
  public readonly out_channels: bigint,
  public readonly kernel_size: [bigint, bigint],
  public readonly stride: [bigint, bigint],
  public readonly padding: [bigint, bigint],
  public readonly output_padding: [bigint, bigint],
  public readonly groups: bigint,
  public readonly bias: boolean,
  public readonly dilation: bigint,
  public readonly padding_mode: Padding_mode,
) {
}

static build(options: Map<string, any>): ConvTranspose2d {
  return new ConvTranspose2d(
    options.get(ConvTranspose2dOptions.In_channels),
  options.get(ConvTranspose2dOptions.Out_channels),
  [  options.get(ConvTranspose2dOptions.Kernel_size[0]), options.get(ConvTranspose2dOptions.Kernel_size)[1]],
  [  options.get(ConvTranspose2dOptions.Stride[0]), options.get(ConvTranspose2dOptions.Stride)[1]],
  [  options.get(ConvTranspose2dOptions.Padding[0]), options.get(ConvTranspose2dOptions.Padding)[1]],
  [  options.get(ConvTranspose2dOptions.Output_padding[0]), options.get(ConvTranspose2dOptions.Output_padding)[1]],
  options.get(ConvTranspose2dOptions.Groups),
  options.get(ConvTranspose2dOptions.Bias),
  options.get(ConvTranspose2dOptions.Dilation),
  getPadding_mode(options.get(ConvTranspose2dOptions.Padding_mode))
  );

  }

  public initCode(): string{
    return `ConvTranspose2d(in_channels=${this.in_channels}, out_channels=${this.out_channels}, kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, output_padding=${this.output_padding}, groups=${this.groups}, bias=${this.bias}, dilation=${this.dilation}, padding_mode=${this.padding_mode})`;
  }
}
