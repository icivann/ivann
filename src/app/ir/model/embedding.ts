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
    return `Embedding(num_embedding=${this.NumEmbeddings}, embedding_dim=${this.EmbeddingDim}, padding_idx= ${this.PaddingIdx}, max_norm= ${this.MaxNorm}, norm_type= ${this.NormType}, scale_grad_by_freq= ${this.ScaleGradByFreq}, sparse= ${this.Sparse}, weight= ${this.Weight})`;
  }
}
