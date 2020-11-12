import { ReLUOptions } from '@/nodes/model/Relu';
import { nodeName } from '@/app/ir/irCommon';

export default class ReLU {
  constructor(
    public readonly name: string,
    public readonly inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): ReLU {
    return new ReLU(
      options.get(nodeName),
      options.get(ReLUOptions.Inplace),
    );
  }

  public initCode(): string {
    return `ReLU(inplace=${this.inplace})`;
  }
}
