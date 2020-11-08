import { ConvTranspose1dOptions } from '@/nodes/pytorch model/Convtranspose1dBaklava';

enum PaddingMode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPaddingMode(s: string): PaddingMode {
  return PaddingMode[s as keyof typeof PaddingMode];
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
  public readonly padding_mode: PaddingMode,
  ) {
  }

  static build(options: Map<string, any>): ConvTranspose1d {
    return new ConvTranspose1d(
      options.get(ConvTranspose1dOptions.InChannels),
      options.get(ConvTranspose1dOptions.OutChannels),
      [options.get(ConvTranspose1dOptions.KernelSize)[0]],
      [options.get(ConvTranspose1dOptions.Stride)[0]],
      [options.get(ConvTranspose1dOptions.Padding)[0]],
      [options.get(ConvTranspose1dOptions.OutputPadding)[0]],
      options.get(ConvTranspose1dOptions.Groups),
      options.get(ConvTranspose1dOptions.Bias),
      [options.get(ConvTranspose1dOptions.Dilation)[0]],
      getPaddingMode(options.get(ConvTranspose1dOptions.PaddingMode)),
    );
  }

  public initCode(): string {
    return `ConvTranspose1d(in_channels=${this.in_channels}, out_channels=${this.out_channels}, kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, output_padding=${this.output_padding}, groups=${this.groups}, bias=${this.bias}, dilation=${this.dilation}, padding_mode=${this.padding_mode})`;
  }
}
