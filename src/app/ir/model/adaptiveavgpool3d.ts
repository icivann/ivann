import { AdaptiveAvgPool3dOptions } from '@/nodes/model/Adaptiveavgpool3d';
import { nodeName } from '@/app/ir/irCommon';

export default class AdaptiveAvgPool3d {
  constructor(
  public readonly name: string,
  public readonly OutputSize: [bigint, bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): AdaptiveAvgPool3d {
    return new AdaptiveAvgPool3d(

      options.get(nodeName),
      [options.get(AdaptiveAvgPool3dOptions.OutputSize)[0], options.get(AdaptiveAvgPool3dOptions.OutputSize)[1], options.get(AdaptiveAvgPool3dOptions.OutputSize)[2]],
    );
  }

  public initCode(): string {
    return `AdaptiveAvgPool3d(OutputSize=(${this.OutputSize}))`;
  }
}
