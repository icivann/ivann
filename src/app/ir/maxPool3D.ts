import { UUID } from '@/app/util';
import { Padding } from '@/app/ir/irCommon';
import MaxPool from '@/app/ir/maxPool';

export default class MaxPool3D extends MaxPool {
  constructor(
        public readonly outputs: Set<UUID>,
        public readonly input: UUID,
        public padding: Padding,
        public readonly kernel: [bigint, bigint, bigint],
        public readonly stride: [bigint, bigint, bigint],
  ) {
    super();
  }
}
