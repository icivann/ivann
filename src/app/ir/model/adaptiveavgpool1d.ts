import { AdaptiveAvgPool1dOptions } from '@/nodes/model/Adaptiveavgpool1d';
import { nodeName } from '@/app/ir/irCommon';

export default class AdaptiveAvgPool1d {
  constructor(
  public readonly name: string,
  public readonly OutputSize: [bigint],
  ) {
  }

  static build(options: Map<string, any>): AdaptiveAvgPool1d {
    return new AdaptiveAvgPool1d(

      options.get(nodeName),
      [options.get(AdaptiveAvgPool1dOptions.OutputSize)[0]],
    );
  }

  public initCode(): string {
    return `AdaptiveAvgPool1d(output_size= ${this.OutputSize})`;
  }
}
