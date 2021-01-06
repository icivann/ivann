import { AdaptiveAvgPool2dOptions } from '@/nodes/model/Adaptiveavgpool2d';
import { nodeName } from '@/app/ir/irCommon';

export default class AdaptiveAvgPool2d {
  constructor(
  public readonly name: string,
  public readonly OutputSize: [bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): AdaptiveAvgPool2d {
    return new AdaptiveAvgPool2d(

      options.get(nodeName),
      [options.get(AdaptiveAvgPool2dOptions.OutputSize)[0], options.get(AdaptiveAvgPool2dOptions.OutputSize)[1]],
    );
  }

  public initCode(): string {
    return `AdaptiveAvgPool2d(OutputSize=(${this.OutputSize}))`;
  }
}
