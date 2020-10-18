import { Padding } from '@/app/ir/irCommon';

export default class MaxPool1D {
  constructor(
        public padding: Padding,
        public readonly kernel: [bigint],
        public readonly stride: [bigint],
  ) {
  }
}
