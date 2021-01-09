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
    return `MultiheadAttention(embed_dim=${this.EmbedDim}, num_heads=${this.NumHeads}, dropout=${this.Dropout}, bias=${this.Bias}, add_bias_k_v=${this.AddBiasKv}, add_zero_attn=${this.AddZeroAttn}, k_dim=${this.Kdim}, v_dim=${this.Vdim})`;
  }
}
