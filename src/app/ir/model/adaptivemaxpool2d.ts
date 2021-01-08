import { AdaptiveMaxPool2dOptions } from '@/nodes/model/Adaptivemaxpool2d';
import { nodeName } from '@/app/ir/irCommon';

export default class AdaptiveMaxPool2d {
  constructor(
  public readonly name: string,
  public readonly OutputSize: [bigint, bigint],
  public readonly ReturnIndices: boolean,
  ) {
  }

  static build(options: Map<string, any>): AdaptiveMaxPool2d {
    return new AdaptiveMaxPool2d(

      options.get(nodeName),
      [options.get(AdaptiveMaxPool2dOptions.OutputSize)[0], options.get(AdaptiveMaxPool2dOptions.OutputSize)[1]],
      options.get(AdaptiveMaxPool2dOptions.ReturnIndices),
    );
  }

  public initCode(): string {
    return `AdaptiveMaxPool2d(OutputSize=(${this.OutputSize}), ReturnIndices=(${this.ReturnIndices}))`;
  }
}
