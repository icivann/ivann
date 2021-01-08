import { LinearOptions } from '@/nodes/model/Linear';
import { nodeName } from '@/app/ir/irCommon';

export default class Linear {
  constructor(
    public readonly name: string,
    public readonly in_features: bigint,
    public readonly out_features: bigint,
    public readonly bias: boolean,
  ) {
  }

  static build(options: Map<string, any>): Linear {
    return new Linear(
      options.get(nodeName),
      options.get(LinearOptions.InFeatures),
      options.get(LinearOptions.OutFeatures),
      options.get(LinearOptions.Bias),
    );
  }

  public initCode(): string {
    return `Linear(in_features=${this.in_features}, out_features=${this.out_features}, bias=${this.bias})`;
  }
}
