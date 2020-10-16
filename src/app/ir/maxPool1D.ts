import { UUID } from '@/app/util';
import { Padding } from '@/app/ir/irCommon';
import MaxPool from '@/app/ir/MaxPool';

export default class MaxPool1D extends MaxPool {
  constructor(
        public readonly outputs: Set<UUID>,
        public readonly input: UUID,
        public padding: Padding,
        public readonly kernel: [bigint],
        public readonly stride: [bigint],
  ) {
    super();
  }

  public code(): string {
    return 'TOOD';
  }
}
