import { SigmoidOptions } from '@/nodes/model/Sigmoid';
import { nodeName } from '@/app/ir/irCommon';

export default class Sigmoid {
  constructor(
  public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Sigmoid {
    return new Sigmoid(

      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'Sigmoid()';
  }
}
