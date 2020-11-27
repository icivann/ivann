import { GELUOptions } from '@/nodes/model/Gelu';
import { nodeName } from '@/app/ir/irCommon';

export default class GELU {
  constructor(
  public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): GELU {
    return new GELU(

      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'GELU()';
  }
}
