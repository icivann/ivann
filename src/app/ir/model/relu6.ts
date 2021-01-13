import { ReLU6Options } from '@/nodes/model/Relu6';
import { nodeName } from '@/app/ir/irCommon';

export default class ReLU6 {
  constructor(
  public readonly name: string,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): ReLU6 {
    return new ReLU6(

      options.get(nodeName),
      options.get(ReLU6Options.Inplace),
    );
  }

  public initCode(): string {
    return `ReLU6(inplace=(${this.Inplace}))`;
  }
}
