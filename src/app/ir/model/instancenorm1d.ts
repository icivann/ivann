import { InstanceNorm1dOptions } from '@/nodes/model/Instancenorm1d';
import { nodeName } from '@/app/ir/irCommon';

export default class InstanceNorm1d {
  constructor(
  public readonly name: string,
  public readonly NumFeatures: bigint,
  public readonly Eps: number,
  public readonly Momentum: number,
  public readonly Affine: boolean,
  public readonly TrackRunningStats: boolean,
  ) {
  }

  static build(options: Map<string, any>): InstanceNorm1d {
    return new InstanceNorm1d(

      options.get(nodeName),
      options.get(InstanceNorm1dOptions.NumFeatures),
      options.get(InstanceNorm1dOptions.Eps),
      options.get(InstanceNorm1dOptions.Momentum),
      options.get(InstanceNorm1dOptions.Affine),
      options.get(InstanceNorm1dOptions.TrackRunningStats),
    );
  }

  public initCode(): string {
    return `InstanceNorm1d(NumFeatures= ${this.NumFeatures}, Eps= ${this.Eps}, Momentum= ${this.Momentum}, Affine= ${this.Affine}, TrackRunningStats= ${this.TrackRunningStats})`;
  }
}
