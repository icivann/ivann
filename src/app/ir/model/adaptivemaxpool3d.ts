import { AdaptiveMaxPool3dOptions } from '@/nodes/model/Adaptivemaxpool3d';
import { nodeName } from '@/app/ir/irCommon';

export default class AdaptiveMaxPool3d {
  constructor(
  public readonly name: string,
  public readonly OutputSize: [bigint, bigint, bigint],
  public readonly ReturnIndices: boolean,
  ) {
  }

  static build(options: Map<string, any>): AdaptiveMaxPool3d {
    return new AdaptiveMaxPool3d(

      options.get(nodeName),
      [options.get(AdaptiveMaxPool3dOptions.OutputSize)[0], options.get(AdaptiveMaxPool3dOptions.OutputSize)[1], options.get(AdaptiveMaxPool3dOptions.OutputSize)[2]],
      options.get(AdaptiveMaxPool3dOptions.ReturnIndices),
    );
  }

  public initCode(): string {
    return `AdaptiveMaxPool3d(OutputSize=, (${this.OutputSize}), ReturnIndices=, (${this.ReturnIndices}))`;
  }
}
