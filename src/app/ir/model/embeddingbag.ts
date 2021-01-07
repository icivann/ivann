import { EmbeddingBagOptions } from '@/nodes/model/Embeddingbag';
import { nodeName, Mode, getMode } from '@/app/ir/irCommon';

export default class EmbeddingBag {
  constructor(
  public readonly name: string,
  public readonly NumEmbeddings: bigint,
  public readonly EmbeddingDim: bigint,
  public readonly MaxNorm: [bigint],
  public readonly NormType: number,
  public readonly ScaleGradByFreq: boolean,
  public readonly Mode: Mode,
  public readonly Sparse: boolean,
  public readonly Weight: [bigint],
  public readonly IncludeLastOffset: boolean,
  ) {
  }

  static build(options: Map<string, any>): EmbeddingBag {
    return new EmbeddingBag(

      options.get(nodeName),
      options.get(EmbeddingBagOptions.NumEmbeddings),
      options.get(EmbeddingBagOptions.EmbeddingDim),
      [options.get(EmbeddingBagOptions.MaxNorm)[0]],
      options.get(EmbeddingBagOptions.NormType),
      options.get(EmbeddingBagOptions.ScaleGradByFreq),
      getMode(options.get(EmbeddingBagOptions.Mode)),
      options.get(EmbeddingBagOptions.Sparse),
      [options.get(EmbeddingBagOptions.Weight)[0]],
      options.get(EmbeddingBagOptions.IncludeLastOffset),
    );
  }

  public initCode(): string {
    return `EmbeddingBag(NumEmbeddings= ${this.NumEmbeddings}, EmbeddingDim= ${this.EmbeddingDim}, MaxNorm= ${this.MaxNorm}, NormType= ${this.NormType}, ScaleGradByFreq= ${this.ScaleGradByFreq}, Mode= ${this.Mode}, Sparse= ${this.Sparse}, weight= ${this.Weight}, IncludeLastOffset= ${this.IncludeLastOffset})`;
  }
}
