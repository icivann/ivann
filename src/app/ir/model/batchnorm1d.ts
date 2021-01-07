import { BatchNorm1dOptions } from '@/nodes/model/Batchnorm1d';
import { nodeName } from '@/app/ir/irCommon';

export default class BatchNorm1d {
  constructor(
  public readonly name: string,
  public readonly NumFeatures: bigint,
  public readonly Eps: number,
  public readonly Momentum: number,
  public readonly Affine: boolean,
  public readonly TrackRunningStats: boolean,
  ) {
  }

  static build(options: Map<string, any>): BatchNorm1d {
    return new BatchNorm1d(

      options.get(nodeName),
      options.get(BatchNorm1dOptions.NumFeatures),
      options.get(BatchNorm1dOptions.Eps),
      options.get(BatchNorm1dOptions.Momentum),
      options.get(BatchNorm1dOptions.Affine),
      options.get(BatchNorm1dOptions.TrackRunningStats),
    );
  }

  public initCode(): string {
    return `BatchNorm1d(NumFeatures=${this.NumFeatures}, Eps=${this.Eps}, Momentum=${this.Momentum}, Affine=${this.Affine}, TrackRunningStats=${this.TrackRunningStats})`;
  }
}
