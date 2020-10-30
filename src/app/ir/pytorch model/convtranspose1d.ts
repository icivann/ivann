import { ConvTranspose1dOptions } from '@/nodes/pytorch model/ConvTranspose1dBaklava';

enum Padding_mode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPadding_mode(s: string) : Padding_mode {
  return Padding_mode[s as keyof typeof Padding_mode];
}

export default class ConvTranspose1d {
constructor(
  public readonly in_channels: bigint,
  public readonly out_channels: bigint,
  public readonly kernel_size: [bigint],
  public readonly stride: [bigint],
  public readonly padding: [bigint],
  public readonly output_padding: [bigint],
  public readonly groups: bigint,
  public readonly bias: boolean,
  public readonly dilation: [bigint],
  public readonly padding_mode: Padding_mode,
) {
}

static build(options: Map<string, any>): ConvTranspose1d {
  return new ConvTranspose1d(
    options.get(ConvTranspose1dOptions.In_channels),
  options.get(ConvTranspose1dOptions.Out_channels),
  [options.get(ConvTranspose1dOptions.Kernel_size)[0]],
  [options.get(ConvTranspose1dOptions.Stride)[0]],
  [options.get(ConvTranspose1dOptions.Padding)[0]],
  [options.get(ConvTranspose1dOptions.Output_padding)[0]],
  options.get(ConvTranspose1dOptions.Groups),
  options.get(ConvTranspose1dOptions.Bias),
  [options.get(ConvTranspose1dOptions.Dilation)[0]],
  getPadding_mode(options.get(ConvTranspose1dOptions.Padding_mode))
  );

  }

  public initCode(): string{
    return `ConvTranspose1d(in_channels=${this.in_channels}, out_channels=${this.out_channels}, kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, output_padding=${this.output_padding}, groups=${this.groups}, bias=${this.bias}, dilation=${this.dilation}, padding_mode=${this.padding_mode})`;
  }
}
