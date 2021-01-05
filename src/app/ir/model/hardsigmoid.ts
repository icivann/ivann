import { HardsigmoidOptions } from '@/nodes/model/Hardsigmoid';
import { nodeName } from '@/app/ir/irCommon';

export default class Hardsigmoid {
  constructor(
  public readonly name: string,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): Hardsigmoid {
    return new Hardsigmoid(

      options.get(nodeName),
      options.get(HardsigmoidOptions.Inplace),
    );
  }

  public initCode(): string {
    return `Hardsigmoid(Inplace= ${this.Inplace})`;
  }
}
