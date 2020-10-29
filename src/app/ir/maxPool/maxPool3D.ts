import { getPadding, Padding } from '@/app/ir/irCommon';

export enum MaxPool3DOptions{
  KernelSize = 'Kernel Size',
  Stride = 'Stride',
  Padding = 'Padding'
}

export default class MaxPool3D {
  constructor(
        public padding: Padding,
        public readonly kernel: [bigint, bigint, bigint],
        public readonly stride: [bigint, bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): MaxPool3D {
    return new MaxPool3D(
      getPadding(options.get(MaxPool3DOptions.Padding)),
      [options.get(MaxPool3DOptions.KernelSize)[0],
        options.get(MaxPool3DOptions.KernelSize)[1],
        options.get(MaxPool3DOptions.KernelSize)[2]],
      [options.get(MaxPool3DOptions.Stride)[0],
        options.get(MaxPool3DOptions.Stride)[1],
        options.get(MaxPool3DOptions.Stride)[2]],
    );
  }
}
