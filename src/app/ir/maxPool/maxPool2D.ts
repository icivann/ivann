import { Padding } from '@/app/ir/irCommon';

export default class MaxPool2D {
  constructor(
        public padding: Padding,
        public readonly kernel: [bigint, bigint],
        public readonly stride: [bigint, bigint],
  ) {
  }
}
