import { RNNBaseOptions } from '@/nodes/model/Rnnbase';
import { nodeName, Mode, getMode } from '@/app/ir/irCommon';

export default class RNNBase {
  constructor(
  public readonly name: string,
  public readonly Mode: Mode,
  public readonly InputSize: bigint,
  public readonly HiddenSize: bigint,
  public readonly NumLayers: bigint,
  public readonly Bias: boolean,
  public readonly BatchFirst: boolean,
  public readonly Dropout: number,
  public readonly Bidirectional: boolean,
  ) {
  }

  static build(options: Map<string, any>): RNNBase {
    return new RNNBase(

      options.get(nodeName),
      getMode(options.get(RNNBaseOptions.Mode)),
      options.get(RNNBaseOptions.InputSize),
      options.get(RNNBaseOptions.HiddenSize),
      options.get(RNNBaseOptions.NumLayers),
      options.get(RNNBaseOptions.Bias),
      options.get(RNNBaseOptions.BatchFirst),
      options.get(RNNBaseOptions.Dropout),
      options.get(RNNBaseOptions.Bidirectional),
    );
  }

  public initCode(): string {
    return `RNNBase(mode=${this.Mode}, input_size=${this.InputSize}, hidden_size=${this.HiddenSize}, num_layers=${this.NumLayers}, bias=${this.Bias}, batch_first=${this.BatchFirst}, dropout=${this.Dropout}, bidirectional=${this.Bidirectional})`;
  }
}
