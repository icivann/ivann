import { EmbeddingOptions } from '@/nodes/model/Embedding';
import { nodeName } from '@/app/ir/irCommon';

export default class Embedding {
  constructor(
  public readonly name: string,
  public readonly NumEmbeddings: bigint,
  public readonly EmbeddingDim: bigint,
  public readonly PaddingIdx: [bigint],
  public readonly MaxNorm: [bigint],
  public readonly NormType: number,
  public readonly ScaleGradByFreq: boolean,
  public readonly Sparse: boolean,
  public readonly Weight: [bigint],
  ) {
  }

  static build(options: Map<string, any>): Embedding {
    return new Embedding(

      options.get(nodeName),
      options.get(EmbeddingOptions.NumEmbeddings),
      options.get(EmbeddingOptions.EmbeddingDim),
      [options.get(EmbeddingOptions.PaddingIdx)[0]],
      [options.get(EmbeddingOptions.MaxNorm)[0]],
      options.get(EmbeddingOptions.NormType),
      options.get(EmbeddingOptions.ScaleGradByFreq),
      options.get(EmbeddingOptions.Sparse),
      [options.get(EmbeddingOptions.Weight)[0]],
    );
  }

  public initCode(): string {
    return `Embedding(NumEmbeddings=, ${this.NumEmbeddings}, EmbeddingDim=, ${this.EmbeddingDim}, PaddingIdx=, ${this.PaddingIdx}, MaxNorm=, ${this.MaxNorm}, NormType=, ${this.NormType}, ScaleGradByFreq=, ${this.ScaleGradByFreq}, Sparse=, ${this.Sparse}, Weight=, ${this.Weight})`;
  }
}
