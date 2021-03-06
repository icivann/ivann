import { TransformerDecoderLayerOptions } from '@/nodes/model/Transformerdecoderlayer';
import { nodeName, Activation, getActivation } from '@/app/ir/irCommon';

export default class TransformerDecoderLayer {
  constructor(
  public readonly name: string,
  public readonly DModel: bigint,
  public readonly Nhead: bigint,
  public readonly DimFeedforward: bigint,
  public readonly Dropout: number,
  public readonly Activation: Activation,
  ) {
  }

  static build(options: Map<string, any>): TransformerDecoderLayer {
    return new TransformerDecoderLayer(

      options.get(nodeName),
      options.get(TransformerDecoderLayerOptions.DModel),
      options.get(TransformerDecoderLayerOptions.Nhead),
      options.get(TransformerDecoderLayerOptions.DimFeedforward),
      options.get(TransformerDecoderLayerOptions.Dropout),
      getActivation(options.get(TransformerDecoderLayerOptions.Activation)),
    );
  }

  public initCode(): string {
    return `TransformerDecoderLayer(d_model=${this.DModel}, nhead=${this.Nhead}, dim_feedforward=${this.DimFeedforward}, dropout=${this.Dropout}, activation=${this.Activation})`;
  }
}
