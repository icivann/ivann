import { TransformerOptions } from '@/nodes/model/Transformer';
import { Activation, getActivation, nodeName } from '@/app/ir/irCommon';

export default class Transformer {
  constructor(
    public readonly name: string,
    public readonly d_model: bigint,
    public readonly nhead: bigint,
    public readonly num_encoder_layers: bigint,
    public readonly num_decoder_layers: bigint,
    public readonly dim_feedforward: bigint,
    public readonly dropout: number,
    public readonly activation: Activation,
    public readonly custom_encoder: [bigint],
    public readonly custom_decoder: [bigint],
  ) {
  }

  static build(options: Map<string, any>): Transformer {
    return new Transformer(
      options.get(nodeName),
      options.get(TransformerOptions.DModel),
      options.get(TransformerOptions.Nhead),
      options.get(TransformerOptions.NumEncoderLayers),
      options.get(TransformerOptions.NumDecoderLayers),
      options.get(TransformerOptions.DimFeedforward),
      options.get(TransformerOptions.Dropout),
      getActivation(options.get(TransformerOptions.Activation)),
      [options.get(TransformerOptions.CustomEncoder)[0]],
      [options.get(TransformerOptions.CustomDecoder)[0]],
    );
  }

  public initCode(): string {
    return `Transformer(d_model=, ${this.d_model}, nhead=, ${this.nhead}, num_encoder_layers=, ${this.num_encoder_layers}, num_decoder_layers=, ${this.num_decoder_layers}, dim_feedforward=, ${this.dim_feedforward}, dropout=, ${this.dropout}, activation=, ${this.activation}, custom_encoder=, ${this.custom_encoder}, custom_decoder=, ${this.custom_decoder})`;
  }
}
