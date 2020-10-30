import { MaxPool3dOptions } from '@/nodes/pytorch model/MaxPool3dBaklava';

export default class MaxPool3d {
constructor(
  public readonly kernel_size: [bigint, bigint, bigint],
  public readonly stride: [bigint, bigint, bigint],
  public readonly padding: [bigint, bigint, bigint],
  public readonly dilation: [bigint, bigint, bigint],
  public readonly return_indices: boolean,
  public readonly ceil_mode: boolean,
) {
}
 
static build(options: Map<string, any>): MaxPool3d {
  return new MaxPool3d(
    [  options.get(MaxPool3dOptions.Kernel_size[0]),   options.get(MaxPool3dOptions.Kernel_size[1]), options.get(MaxPool3dOptions.Kernel_size)[2]], 
  [  options.get(MaxPool3dOptions.Stride[0]),   options.get(MaxPool3dOptions.Stride[1]), options.get(MaxPool3dOptions.Stride)[2]], 
  [  options.get(MaxPool3dOptions.Padding[0]),   options.get(MaxPool3dOptions.Padding[1]), options.get(MaxPool3dOptions.Padding)[2]], 
  [  options.get(MaxPool3dOptions.Dilation[0]),   options.get(MaxPool3dOptions.Dilation[1]), options.get(MaxPool3dOptions.Dilation)[2]], 
  options.get(MaxPool3dOptions.Return_indices),
  options.get(MaxPool3dOptions.Ceil_mode),
  );
  
  }
  
  public initCode(): string{
    return `MaxPool3d(kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, dilation=${this.dilation}, return_indices=${this.return_indices}, ceil_mode=${this.ceil_mode})`;
  }
}
  