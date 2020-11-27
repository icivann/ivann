import { TanhOptions } from '@/nodes/model/Tanh';
import { nodeName } from '@/app/ir/irCommon';

export default class Tanh {
  constructor(
  public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Tanh {
    return new Tanh(

      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'Tanh()';
  }
}
