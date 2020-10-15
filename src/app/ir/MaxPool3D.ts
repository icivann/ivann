import MaxPool from '@/app/ir/MaxPool';
import { UUID } from '@/app/util';
import { Padding } from '@/app/ir/irCommon';

class MaxPool3D extends MaxPool {
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

export default MaxPool3D;
