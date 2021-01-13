import { RNNOptions } from '@/nodes/model/Rnn';
import { nodeName, Nonlinearity, getNonlinearity } from '@/app/ir/irCommon';

export default class RNN {
  constructor(
  public readonly name: string,
  public readonly InputSize: bigint,
  public readonly HiddenSize: bigint,
  public readonly NumLayers: bigint,
  public readonly Nonlinearity: Nonlinearity,
  public readonly Bias: boolean,
  public readonly BatchFirst: boolean,
  public readonly Dropout: number,
  public readonly Bidirectional: boolean,
  ) {
  }

  static build(options: Map<string, any>): RNN {
    return new RNN(

      options.get(nodeName),
      options.get(RNNOptions.InputSize),
      options.get(RNNOptions.HiddenSize),
      options.get(RNNOptions.NumLayers),
      getNonlinearity(options.get(RNNOptions.Nonlinearity)),
      options.get(RNNOptions.Bias),
      options.get(RNNOptions.BatchFirst),
      options.get(RNNOptions.Dropout),
      options.get(RNNOptions.Bidirectional),
    );
  }

  public initCode(): string {
    return `RNN(input_size=${this.InputSize}, hidden_size=${this.HiddenSize}, num_layers=${this.NumLayers}, nonlinearity=${this.Nonlinearity}, bias=${this.Bias}, batch_first=${this.BatchFirst}, dropout=${this.Dropout}, bidirectional=${this.Bidirectional})`;
  }
}
