import { TransformerEncoderLayerOptions } from '@/nodes/model/Transformerencoderlayer';
import { nodeName, Activation, getActivation } from '@/app/ir/irCommon';

export default class TransformerEncoderLayer {
  constructor(
  public readonly name: string,
  public readonly DModel: bigint,
  public readonly Nhead: bigint,
  public readonly DimFeedforward: bigint,
  public readonly Dropout: number,
  public readonly Activation: Activation,
  ) {
  }

  static build(options: Map<string, any>): TransformerEncoderLayer {
    return new TransformerEncoderLayer(

      options.get(nodeName),
      options.get(TransformerEncoderLayerOptions.DModel),
      options.get(TransformerEncoderLayerOptions.Nhead),
      options.get(TransformerEncoderLayerOptions.DimFeedforward),
      options.get(TransformerEncoderLayerOptions.Dropout),
      getActivation(options.get(TransformerEncoderLayerOptions.Activation)),
    );
  }

  public initCode(): string {
    return `TransformerEncoderLayer(d_model=${this.DModel}, nhead=${this.Nhead}, dim_feedforward=${this.DimFeedforward}, dropout=${this.Dropout}, activation=${this.Activation})`;
  }
}
