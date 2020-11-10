import { ConvTranspose3dOptions } from '@/nodes/model/Convtranspose3d';
import { PaddingMode } from '@/app/ir/irCommon';

function getPaddingMode(s: string): PaddingMode {
  return PaddingMode[s as keyof typeof PaddingMode];
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
  public readonly padding_mode: PaddingMode,
  ) {
  }

  static build(options: Map<string, any>): ConvTranspose3d {
    return new ConvTranspose3d(
      options.get(ConvTranspose3dOptions.InChannels),
      options.get(ConvTranspose3dOptions.OutChannels),
      [options.get(ConvTranspose3dOptions.KernelSize[0]),
        options.get(ConvTranspose3dOptions.KernelSize[1]),
        options.get(ConvTranspose3dOptions.KernelSize)[2]],
      [options.get(ConvTranspose3dOptions.Stride[0]),
        options.get(ConvTranspose3dOptions.Stride[1]),
        options.get(ConvTranspose3dOptions.Stride)[2]],
      [options.get(ConvTranspose3dOptions.Padding[0]),
        options.get(ConvTranspose3dOptions.Padding[1]),
        options.get(ConvTranspose3dOptions.Padding)[2]],
      [options.get(ConvTranspose3dOptions.OutputPadding[0]),
        options.get(ConvTranspose3dOptions.OutputPadding[1]),
        options.get(ConvTranspose3dOptions.OutputPadding)[2]],
      options.get(ConvTranspose3dOptions.Groups),
      options.get(ConvTranspose3dOptions.Bias),
      [options.get(ConvTranspose3dOptions.Dilation[0]),
        options.get(ConvTranspose3dOptions.Dilation[1]),
        options.get(ConvTranspose3dOptions.Dilation)[2]],
      getPaddingMode(options.get(ConvTranspose3dOptions.PaddingMode)),
    );
  }

  public initCode(): string {
    return `ConvTranspose3d(in_channels=${this.in_channels}, out_channels=${this.out_channels}, kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, output_padding=${this.output_padding}, groups=${this.groups}, bias=${this.bias}, dilation=${this.dilation}, padding_mode=${this.padding_mode})`;
  }
}
