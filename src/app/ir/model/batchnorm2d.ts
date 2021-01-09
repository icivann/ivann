import { BatchNorm2dOptions } from '@/nodes/model/Batchnorm2d';
import { nodeName } from '@/app/ir/irCommon';

export default class BatchNorm2d {
  constructor(
  public readonly name: string,
  public readonly NumFeatures: bigint,
  public readonly Eps: number,
  public readonly Momentum: number,
  public readonly Affine: boolean,
  public readonly TrackRunningStats: boolean,
  ) {
  }

  static build(options: Map<string, any>): BatchNorm2d {
    return new BatchNorm2d(

      options.get(nodeName),
      options.get(BatchNorm2dOptions.NumFeatures),
      options.get(BatchNorm2dOptions.Eps),
      options.get(BatchNorm2dOptions.Momentum),
      options.get(BatchNorm2dOptions.Affine),
      options.get(BatchNorm2dOptions.TrackRunningStats),
    );
  }

  public initCode(): string {
    return `BatchNorm2d(num_features=(${this.NumFeatures}), eps=(${this.Eps}), momentum=(${this.Momentum}), affine=(${this.Affine}), trace_running_stats=(${this.TrackRunningStats}))`;
  }
}
