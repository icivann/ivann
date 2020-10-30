import { ConvTranspose3dOptions } from '@/nodes/pytorch model/ConvTranspose3dBaklava';

enum Padding_mode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPadding_mode(s: string) : Padding_mode {
  return Padding_mode[s as keyof typeof Padding_mode];
}
export default class ConvTranspose3d {
constructor(
  public readonly in_channels: bigint,
  public readonly out_channels: bigint,
  public readonly kernel_size: [bigint, bigint, bigint],
  public readonly stride: [bigint, bigint, bigint],
  public readonly padding: [bigint, bigint, bigint],
  public readonly output_padding: [bigint, bigint, bigint],
  public readonly groups: bigint,
  public readonly bias: boolean,
  public readonly dilation: [bigint, bigint, bigint],
  public readonly padding_mode: Padding_mode,
) {
}

static build(options: Map<string, any>): ConvTranspose3d {
  return new ConvTranspose3d(
    options.get(ConvTranspose3dOptions.In_channels),
  options.get(ConvTranspose3dOptions.Out_channels),
  [  options.get(ConvTranspose3dOptions.Kernel_size[0]),   options.get(ConvTranspose3dOptions.Kernel_size[1]), options.get(ConvTranspose3dOptions.Kernel_size)[2]],
  [  options.get(ConvTranspose3dOptions.Stride[0]),   options.get(ConvTranspose3dOptions.Stride[1]), options.get(ConvTranspose3dOptions.Stride)[2]],
  [  options.get(ConvTranspose3dOptions.Padding[0]),   options.get(ConvTranspose3dOptions.Padding[1]), options.get(ConvTranspose3dOptions.Padding)[2]],
  [  options.get(ConvTranspose3dOptions.Output_padding[0]),   options.get(ConvTranspose3dOptions.Output_padding[1]), options.get(ConvTranspose3dOptions.Output_padding)[2]],
  options.get(ConvTranspose3dOptions.Groups),
  options.get(ConvTranspose3dOptions.Bias),
  [  options.get(ConvTranspose3dOptions.Dilation[0]),   options.get(ConvTranspose3dOptions.Dilation[1]), options.get(ConvTranspose3dOptions.Dilation)[2]],
  getPadding_mode(options.get(ConvTranspose3dOptions.Padding_mode))
  );

  }

  public initCode(): string{
    return `ConvTranspose3d(in_channels=${this.in_channels}, out_channels=${this.out_channels}, kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, output_padding=${this.output_padding}, groups=${this.groups}, bias=${this.bias}, dilation=${this.dilation}, padding_mode=${this.padding_mode})`;
  }
}
