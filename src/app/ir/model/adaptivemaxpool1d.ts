import { AdaptiveMaxPool1dOptions } from '@/nodes/model/Adaptivemaxpool1d';
import { nodeName } from '@/app/ir/irCommon';

export default class AdaptiveMaxPool1d {
  constructor(
  public readonly name: string,
  public readonly OutputSize: [bigint],
  public readonly ReturnIndices: boolean,
  ) {
  }

  static build(options: Map<string, any>): AdaptiveMaxPool1d {
    return new AdaptiveMaxPool1d(

      options.get(nodeName),
      [options.get(AdaptiveMaxPool1dOptions.OutputSize)[0]],
      options.get(AdaptiveMaxPool1dOptions.ReturnIndices),
    );
  }

  public initCode(): string {
    return `AdaptiveMaxPool1d(OutputSize= ${this.OutputSize}, ReturnIndices= ${this.ReturnIndices})`;
  }
}
