import { LSTMOptions } from '@/nodes/model/Lstm';
import { nodeName } from '@/app/ir/irCommon';

export default class LSTM {
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

  static build(options: Map<string, any>): LSTM {
    return new LSTM(

      options.get(nodeName),
      options.get(LSTMOptions.InputSize),
      options.get(LSTMOptions.HiddenSize),
      options.get(LSTMOptions.NumLayers),
      options.get(LSTMOptions.Bias),
      options.get(LSTMOptions.BatchFirst),
      options.get(LSTMOptions.Dropout),
      options.get(LSTMOptions.Bidirectional),
    );
  }

  public initCode(): string {
    return `LSTM(InputSize=${this.InputSize}, HiddenSize=${this.HiddenSize}, NumLayers= ${this.NumLayers}, Bias= ${this.Bias}, BatchFirst= ${this.BatchFirst}, Dropout= ${this.Dropout}, Bidirectional= ${this.Bidirectional})`;
  }
}
