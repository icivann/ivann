import { MultiheadAttentionOptions } from '@/nodes/model/Multiheadattention';
import { nodeName } from '@/app/ir/irCommon';

export default class MultiheadAttention {
  constructor(
  public readonly name: string,
  public readonly EmbedDim: bigint,
  public readonly NumHeads: bigint,
  public readonly Dropout: number,
  public readonly Bias: boolean,
  public readonly AddBiasKv: boolean,
  public readonly AddZeroAttn: boolean,
  public readonly Kdim: bigint,
  public readonly Vdim: bigint,
  ) {
  }

  static build(options: Map<string, any>): MultiheadAttention {
    return new MultiheadAttention(

      options.get(nodeName),
      options.get(MultiheadAttentionOptions.EmbedDim),
      options.get(MultiheadAttentionOptions.NumHeads),
      options.get(MultiheadAttentionOptions.Dropout),
      options.get(MultiheadAttentionOptions.Bias),
      options.get(MultiheadAttentionOptions.AddBiasKv),
      options.get(MultiheadAttentionOptions.AddZeroAttn),
      options.get(MultiheadAttentionOptions.Kdim),
      options.get(MultiheadAttentionOptions.Vdim),
    );
  }

  public initCode(): string {
    return `MultiheadAttention(EmbedDim=${this.EmbedDim}, NumHeads=${this.NumHeads}, Dropout=${this.Dropout}, Bias=${this.Bias}, AddBiasKv=${this.AddBiasKv}, AddZeroAttn=${this.AddZeroAttn}, Kdim=${this.Kdim}, Vdim=${this.Vdim})`;
  }
}
