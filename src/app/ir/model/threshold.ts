import { ThresholdOptions } from '@/nodes/model/Threshold';
import { nodeName } from '@/app/ir/irCommon';

export default class Threshold {
  constructor(
  public readonly name: string,
  public readonly Threshold: number,
  public readonly Value: number,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): Threshold {
    return new Threshold(

      options.get(nodeName),
      options.get(ThresholdOptions.Threshold),
      options.get(ThresholdOptions.Value),
      options.get(ThresholdOptions.Inplace),
    );
  }

  public initCode(): string {
    return `Threshold(Threshold=, ${this.Threshold}, Value=, ${this.Value}, Inplace=, ${this.Inplace})`;
  }
}
