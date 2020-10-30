import { Conv1dOptions } from '@/nodes/pytorch model/Conv1dBaklava';

enum Padding_mode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPadding_mode(s: string) : Padding_mode {
  return Padding_mode[s as keyof typeof Padding_mode];
}

export default class Conv1d {
constructor(
  public readonly in_channels: bigint,
  public readonly out_channels: bigint,
  public readonly kernel_size: [bigint],
  public readonly stride: [bigint],
  public readonly padding: [bigint],
  public readonly dilation: [bigint],
  public readonly groups: bigint,
  public readonly bias: boolean,
  public readonly padding_mode: Padding_mode,
) {
}

static build(options: Map<string, any>): Conv1d {

  return new Conv1d(
    options.get(Conv1dOptions.In_channels),
  options.get(Conv1dOptions.Out_channels),
  [options.get(Conv1dOptions.Kernel_size)[0]],
  [options.get(Conv1dOptions.Stride)[0]],
  [options.get(Conv1dOptions.Padding)[0]],
  [options.get(Conv1dOptions.Dilation)[0]],
  options.get(Conv1dOptions.Groups),
  options.get(Conv1dOptions.Bias),
  getPadding_mode(options.get(Conv1dOptions.Padding_mode))
  );

  }

  public initCode(): string{
    return `Conv1d(in_channels=${this.in_channels}, out_channels=${this.out_channels}, kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, dilation=${this.dilation}, groups=${this.groups}, bias=${this.bias}, padding_mode='${this.padding_mode}')`;
  }
}
