import { SyncBatchNormOptions } from '@/nodes/model/Syncbatchnorm';
import { nodeName } from '@/app/ir/irCommon';

export default class SyncBatchNorm {
  constructor(
  public readonly name: string,
  public readonly NumFeatures: bigint,
  public readonly Eps: number,
  public readonly Momentum: number,
  public readonly Affine: boolean,
  public readonly TrackRunningStats: boolean,
  public readonly ProcessGroup: [bigint],
  ) {
  }

  static build(options: Map<string, any>): SyncBatchNorm {
    return new SyncBatchNorm(

      options.get(nodeName),
      options.get(SyncBatchNormOptions.NumFeatures),
      options.get(SyncBatchNormOptions.Eps),
      options.get(SyncBatchNormOptions.Momentum),
      options.get(SyncBatchNormOptions.Affine),
      options.get(SyncBatchNormOptions.TrackRunningStats),
      [options.get(SyncBatchNormOptions.ProcessGroup)[0]],
    );
  }

  public initCode(): string {
    return `SyncBatchNorm(num_features=${this.NumFeatures}, eps=${this.Eps}, momentum=${this.Momentum}, affine=${this.Affine}, trace_running_stats=${this.TrackRunningStats}, process_group=${this.ProcessGroup})`;
  }
}
