import { HardshrinkOptions } from '@/nodes/model/Hardshrink';
import { nodeName } from '@/app/ir/irCommon';

export default class Hardshrink {
  constructor(
  public readonly name: string,
  public readonly Lambd: number,
  ) {
  }

  static build(options: Map<string, any>): Hardshrink {
    return new Hardshrink(

      options.get(nodeName),
      options.get(HardshrinkOptions.Lambd),
    );
  }

  public initCode(): string {
    return `Hardshrink(lambd=${this.Lambd})`;
  }
}
