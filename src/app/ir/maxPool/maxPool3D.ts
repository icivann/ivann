import { Padding } from '@/app/ir/irCommon';

export default class MaxPool3D {
  constructor(
        public padding: Padding,
        public readonly kernel: [bigint, bigint, bigint],
        public readonly stride: [bigint, bigint, bigint],
  ) {
  }
  public initCode(): string {
    return `MaxPool3d(${this.kernel})`;
  }
}
