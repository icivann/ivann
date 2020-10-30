import { MaxPool2dOptions } from '@/nodes/pytorch model/MaxPool2dBaklava';

export default class MaxPool2d {
constructor(
  public readonly kernel_size: [bigint, bigint],
  public readonly stride: [bigint, bigint],
  public readonly padding: [bigint, bigint],
  public readonly dilation: [bigint, bigint],
  public readonly return_indices: boolean,
  public readonly ceil_mode: boolean,
) {
}
 
static build(options: Map<string, any>): MaxPool2d {
  return new MaxPool2d(
    [  options.get(MaxPool2dOptions.Kernel_size[0]), options.get(MaxPool2dOptions.Kernel_size)[1]], 
  [  options.get(MaxPool2dOptions.Stride[0]), options.get(MaxPool2dOptions.Stride)[1]], 
  [  options.get(MaxPool2dOptions.Padding[0]), options.get(MaxPool2dOptions.Padding)[1]], 
  [  options.get(MaxPool2dOptions.Dilation[0]), options.get(MaxPool2dOptions.Dilation)[1]], 
  options.get(MaxPool2dOptions.Return_indices),
  options.get(MaxPool2dOptions.Ceil_mode),
  );
  
  }
  
  public initCode(): string{
    return `MaxPool2d(kernel_size=${this.kernel_size}, stride=${this.stride}, padding=${this.padding}, dilation=${this.dilation}, return_indices=${this.return_indices}, ceil_mode=${this.ceil_mode})`;
  }
}
  