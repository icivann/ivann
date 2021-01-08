import { GroupNormOptions } from '@/nodes/model/Groupnorm';
import { nodeName } from '@/app/ir/irCommon';

export default class GroupNorm {
  constructor(
  public readonly name: string,
  public readonly NumGroups: bigint,
  public readonly NumChannels: bigint,
  public readonly Eps: number,
  public readonly Affine: boolean,
  ) {
  }

  static build(options: Map<string, any>): GroupNorm {
    return new GroupNorm(

      options.get(nodeName),
      options.get(GroupNormOptions.NumGroups),
      options.get(GroupNormOptions.NumChannels),
      options.get(GroupNormOptions.Eps),
      options.get(GroupNormOptions.Affine),
    );
  }

  public initCode(): string {
    return `GroupNorm(num_groups= ${this.NumGroups}, num_channels= ${this.NumChannels}, eps= ${this.Eps}, affine= ${this.Affine})`;
  }
}
