import { InstanceNorm3dOptions } from '@/nodes/model/Instancenorm3d';
import { nodeName } from '@/app/ir/irCommon';

export default class InstanceNorm3d {
  constructor(
  public readonly name: string,
  public readonly NumFeatures: bigint,
  public readonly Eps: number,
  public readonly Momentum: number,
  public readonly Affine: boolean,
  public readonly TrackRunningStats: boolean,
  ) {
  }

  static build(options: Map<string, any>): InstanceNorm3d {
    return new InstanceNorm3d(

      options.get(nodeName),
      options.get(InstanceNorm3dOptions.NumFeatures),
      options.get(InstanceNorm3dOptions.Eps),
      options.get(InstanceNorm3dOptions.Momentum),
      options.get(InstanceNorm3dOptions.Affine),
      options.get(InstanceNorm3dOptions.TrackRunningStats),
    );
  }

  public initCode(): string {
    return `InstanceNorm3d(NumFeatures= (${this.NumFeatures}), Eps= (${this.Eps}), Momentum= (${this.Momentum}), Affine= (${this.Affine}), TrackRunningStats= (${this.TrackRunningStats}))`;
  }
}
