import { AdaptiveLogSoftmaxWithLossOptions } from '@/nodes/model/Adaptivelogsoftmaxwithloss';
import { nodeName } from '@/app/ir/irCommon';

export default class AdaptiveLogSoftmaxWithLoss {
  constructor(
  public readonly name: string,
  public readonly InFeatures: bigint,
  public readonly NClasses: bigint,
  public readonly DivValue: number,
  public readonly HeadBias: boolean,
  ) {
  }

  static build(options: Map<string, any>): AdaptiveLogSoftmaxWithLoss {
    return new AdaptiveLogSoftmaxWithLoss(

      options.get(nodeName),
      options.get(AdaptiveLogSoftmaxWithLossOptions.InFeatures),
      options.get(AdaptiveLogSoftmaxWithLossOptions.NClasses),
      options.get(AdaptiveLogSoftmaxWithLossOptions.DivValue),
      options.get(AdaptiveLogSoftmaxWithLossOptions.HeadBias),
    );
  }

  public initCode(): string {
    return `AdaptiveLogSoftmaxWithLoss(in_features=${this.InFeatures}, n_classes=${this.NClasses}, div_value=${this.DivValue}, head_bias=${this.HeadBias})`;
  }
}
