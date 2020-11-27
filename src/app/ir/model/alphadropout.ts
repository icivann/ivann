import { AlphaDropoutOptions } from '@/nodes/model/Alphadropout';
import { nodeName } from '@/app/ir/irCommon';

export default class AlphaDropout {
  constructor(
  public readonly name: string,
  public readonly P: number,
  public readonly Inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): AlphaDropout {
    return new AlphaDropout(

      options.get(nodeName),
      options.get(AlphaDropoutOptions.P),
      options.get(AlphaDropoutOptions.Inplace),
    );
  }

  public initCode(): string {
    return `AlphaDropout(P=, ${this.P}, Inplace=, ${this.Inplace})`;
  }
}
