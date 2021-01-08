import { SoftshrinkOptions } from '@/nodes/model/Softshrink';
import { nodeName } from '@/app/ir/irCommon';

export default class Softshrink {
  constructor(
  public readonly name: string,
  public readonly Lambd: number,
  ) {
  }

  static build(options: Map<string, any>): Softshrink {
    return new Softshrink(

      options.get(nodeName),
      options.get(SoftshrinkOptions.Lambd),
    );
  }

  public initCode(): string {
    return `Softshrink(lambd=${this.Lambd})`;
  }
}
