import { CosineSimilarityOptions } from '@/nodes/model/Cosinesimilarity';
import { nodeName } from '@/app/ir/irCommon';

export default class CosineSimilarity {
  constructor(
  public readonly name: string,
  public readonly Dim: bigint,
  public readonly Eps: number,
  ) {
  }

  static build(options: Map<string, any>): CosineSimilarity {
    return new CosineSimilarity(

      options.get(nodeName),
      options.get(CosineSimilarityOptions.Dim),
      options.get(CosineSimilarityOptions.Eps),
    );
  }

  public initCode(): string {
    return `CosineSimilarity(Dim=${this.Dim}, Eps=${this.Eps})`;
  }
}
