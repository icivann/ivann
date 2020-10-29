import { getPadding, Padding } from '@/app/ir/irCommon';
import { MaxPool2DOptions } from '@/nodes/model/pool/MaxPool2D';

export default class MaxPool2D {
  constructor(
        public padding: Padding,
        public readonly kernel: [bigint, bigint],
        public readonly stride: [bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): MaxPool2D {
    if (!(options.size === Object.keys(MaxPool2DOptions).length
      && options.get(MaxPool2DOptions.KernelSize).size === 2
      && options.get(MaxPool2DOptions.Stride).size === 2
    )) {
      console.log('DEAD');
    }
    return new MaxPool2D(
      getPadding(options.get(MaxPool2DOptions.Padding)),
      [options.get(MaxPool2DOptions.KernelSize)[0], options.get(MaxPool2DOptions.KernelSize)[1]],
      [options.get(MaxPool2DOptions.Stride)[0], options.get(MaxPool2DOptions.Stride)[1]],
    );
  }
}
