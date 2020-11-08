import { ConvTranspose2dOptions } from '@/nodes/model/Convtranspose2d';

enum PaddingMode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPaddingMode(s: string): PaddingMode {
  return PaddingMode[s as keyof typeof PaddingMode];
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
  public readonly padding_mode: PaddingMode,
  ) {
  }

  static build(options: Map<string, any>): ConvTranspose2d {
    return new ConvTranspose2d(
      options.get(ConvTranspose2dOptions.InChannels),
      options.get(ConvTranspose2dOptions.OutChannels),
      [options.get(ConvTranspose2dOptions.KernelSize[0]),
        options.get(ConvTranspose2dOptions.KernelSize)[1]],
      [options.get(ConvTranspose2dOptions.Stride[0]),
        options.get(ConvTranspose2dOptions.Stride)[1]],
      [options.get(ConvTranspose2dOptions.Padding[0]),
        options.get(ConvTranspose2dOptions.Padding)[1]],
      [options.get(ConvTranspose2dOptions.OutputPadding[0]),
        options.get(ConvTranspose2dOptions.OutputPadding)[1]],
      options.get(ConvTranspose2dOptions.Groups),
      options.get(ConvTranspose2dOptions.Bias),
      options.get(ConvTranspose2dOptions.Dilation),
      getPaddingMode(options.get(ConvTranspose2dOptions.PaddingMode)),
    );
  }

  public initCode(): string {
    return `ConvTranspose2d(in_channels=${this.in_channels}, out_channels=${this.out_channels}, kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, output_padding=${this.output_padding}, groups=${this.groups}, bias=${this.bias}, dilation=${this.dilation}, padding_mode=${this.padding_mode})`;
  }
}
