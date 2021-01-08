import { LocalResponseNormOptions } from '@/nodes/model/Localresponsenorm';
import { nodeName } from '@/app/ir/irCommon';

export default class LocalResponseNorm {
  constructor(
  public readonly name: string,
  public readonly Size: bigint,
  public readonly Alpha: number,
  public readonly Beta: number,
  public readonly K: number,
  ) {
  }

  static build(options: Map<string, any>): LocalResponseNorm {
    return new LocalResponseNorm(

      options.get(nodeName),
      options.get(LocalResponseNormOptions.Size),
      options.get(LocalResponseNormOptions.Alpha),
      options.get(LocalResponseNormOptions.Beta),
      options.get(LocalResponseNormOptions.K),
    );
  }

  public initCode(): string {
    return `LocalResponseNorm(Size=${this.Size}, Alpha=${this.Alpha}, Beta=${this.Beta}, K=${this.K})`;
  }
}
