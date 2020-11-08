import { Conv1dOptions } from '@/nodes/pytorch model/Conv1dBaklava';

enum PaddingMode {
  zeros ='zeros', reflect ='reflect', replicate ='replicate', circular = 'circular'
}
function getPaddingMode(s: string): PaddingMode {
  return PaddingMode[s as keyof typeof PaddingMode];
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
  public readonly padding_mode: PaddingMode,
  ) {
  }

  static build(options: Map<string, any>): Conv1d {
    return new Conv1d(
      options.get(Conv1dOptions.InChannels),
      options.get(Conv1dOptions.OutChannels),
      [options.get(Conv1dOptions.KernelSize)[0]],
      [options.get(Conv1dOptions.Stride)[0]],
      [options.get(Conv1dOptions.Padding)[0]],
      [options.get(Conv1dOptions.Dilation)[0]],
      options.get(Conv1dOptions.Groups),
      options.get(Conv1dOptions.Bias),
      getPaddingMode(options.get(Conv1dOptions.PaddingMode)),
    );
  }

  public initCode(): string {
    return `Conv1d(in_channels=${this.in_channels}, out_channels=${this.out_channels},
    kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding},
    dilation=${this.dilation}, groups=${this.groups}, bias=${this.bias},
    padding_mode='${this.padding_mode}')`;
  }

  public callCode(params: string[], name: string): string {
    return `${name}(${params.join(', ')})`;
  }
}
