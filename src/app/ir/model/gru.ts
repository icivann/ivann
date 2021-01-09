import { GRUOptions } from '@/nodes/model/Gru';
import { nodeName } from '@/app/ir/irCommon';

export default class GRU {
  constructor(
  public readonly name: string,
  public readonly InputSize: bigint,
  public readonly HiddenSize: bigint,
  public readonly NumLayers: bigint,
  public readonly Bias: boolean,
  public readonly BatchFirst: boolean,
  public readonly Dropout: number,
  public readonly Bidirectional: boolean,
  ) {
  }

  static build(options: Map<string, any>): GRU {
    return new GRU(

      options.get(nodeName),
      options.get(GRUOptions.InputSize),
      options.get(GRUOptions.HiddenSize),
      options.get(GRUOptions.NumLayers),
      options.get(GRUOptions.Bias),
      options.get(GRUOptions.BatchFirst),
      options.get(GRUOptions.Dropout),
      options.get(GRUOptions.Bidirectional),
    );
  }

  public initCode(): string {
    return `GRU(input_size=${this.InputSize}, hidden_size=${this.HiddenSize}, num_layers=${this.NumLayers}, bias=${this.Bias}, batch_first=${this.BatchFirst}, dropout=${this.Dropout}, bidirectional=${this.Bidirectional})`;
  }
}
