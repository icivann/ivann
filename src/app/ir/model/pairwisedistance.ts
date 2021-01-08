import { PairwiseDistanceOptions } from '@/nodes/model/Pairwisedistance';
import { nodeName } from '@/app/ir/irCommon';

export default class PairwiseDistance {
  constructor(
  public readonly name: string,
  public readonly P: number,
  public readonly Eps: number,
  public readonly Keepdim: boolean,
  ) {
  }

  static build(options: Map<string, any>): PairwiseDistance {
    return new PairwiseDistance(

      options.get(nodeName),
      options.get(PairwiseDistanceOptions.P),
      options.get(PairwiseDistanceOptions.Eps),
      options.get(PairwiseDistanceOptions.Keepdim),
    );
  }

  public initCode(): string {
    return `PairwiseDistance(P= ${this.P}, Eps= ${this.Eps}, Keepdim= ${this.Keepdim})`;
  }
}
