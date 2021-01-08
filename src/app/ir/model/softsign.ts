import { SoftsignOptions } from '@/nodes/model/Softsign';
import { nodeName } from '@/app/ir/irCommon';

export default class Softsign {
  constructor(
  public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Softsign {
    return new Softsign(

      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'Softsign()';
  }
}
