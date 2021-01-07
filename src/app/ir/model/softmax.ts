import { SoftmaxOptions } from '@/nodes/model/Softmax';
import { nodeName } from '@/app/ir/irCommon';

export default class Softmax {
  constructor(
  public readonly name: string,
  public readonly dim: [bigint],
  ) {
  }

  static build(options: Map<string, any>): Softmax {
    return new Softmax(
      options.get(nodeName),
      [options.get(SoftmaxOptions.Dim)[0]],
    );
  }

  public initCode(): string {
    return `Softmax(dim=${this.dim})`;
  }
}
