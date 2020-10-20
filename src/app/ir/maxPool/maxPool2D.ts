import { Padding } from '@/app/ir/irCommon';

export default class MaxPool2D {
  constructor(
        public padding: Padding,
        public readonly kernel: [bigint, bigint],
        public readonly stride: [bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): MaxPool2D {
    return new MaxPool2D(
      options.get('Padding'),
      [options.get('Kernel Size')[0], options.get('Kernel Size')[1]],
      [options.get('Stride')[0], options.get('Stride')[1]],
    );
  }
}
