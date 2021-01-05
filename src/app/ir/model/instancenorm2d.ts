import { InstanceNorm2dOptions } from '@/nodes/model/Instancenorm2d';
import { nodeName } from '@/app/ir/irCommon';

export default class InstanceNorm2d {
  constructor(
  public readonly name: string,
  public readonly NumFeatures: bigint,
  public readonly Eps: number,
  public readonly Momentum: number,
  public readonly Affine: boolean,
  public readonly TrackRunningStats: boolean,
  ) {
  }

  static build(options: Map<string, any>): InstanceNorm2d {
    return new InstanceNorm2d(

      options.get(nodeName),
      options.get(InstanceNorm2dOptions.NumFeatures),
      options.get(InstanceNorm2dOptions.Eps),
      options.get(InstanceNorm2dOptions.Momentum),
      options.get(InstanceNorm2dOptions.Affine),
      options.get(InstanceNorm2dOptions.TrackRunningStats),
    );
  }

  public initCode(): string {
    return `InstanceNorm2d(NumFeatures= (${this.NumFeatures}), Eps= (${this.Eps}), Momentum= (${this.Momentum}), Affine= (${this.Affine}), TrackRunningStats= (${this.TrackRunningStats}))`;
  }
}
