import { BatchNorm3dOptions } from '@/nodes/model/Batchnorm3d';
import { nodeName } from '@/app/ir/irCommon';

export default class BatchNorm3d {
  constructor(
  public readonly name: string,
  public readonly NumFeatures: bigint,
  public readonly Eps: number,
  public readonly Momentum: number,
  public readonly Affine: boolean,
  public readonly TrackRunningStats: boolean,
  ) {
  }

  static build(options: Map<string, any>): BatchNorm3d {
    return new BatchNorm3d(

      options.get(nodeName),
      options.get(BatchNorm3dOptions.NumFeatures),
      options.get(BatchNorm3dOptions.Eps),
      options.get(BatchNorm3dOptions.Momentum),
      options.get(BatchNorm3dOptions.Affine),
      options.get(BatchNorm3dOptions.TrackRunningStats),
    );
  }

  public initCode(): string {
    return `BatchNorm3d(num_features=(${this.NumFeatures}), eps=(${this.Eps}), momentum=(${this.Momentum}), affine=(${this.Affine}), trace_running_stats=(${this.TrackRunningStats}))`;
  }
}
