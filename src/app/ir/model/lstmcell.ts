import { LSTMCellOptions } from '@/nodes/model/Lstmcell';
import { nodeName } from '@/app/ir/irCommon';

export default class LSTMCell {
  constructor(
  public readonly name: string,
  public readonly InputSize: bigint,
  public readonly HiddenSize: bigint,
  public readonly Bias: boolean,
  ) {
  }

  static build(options: Map<string, any>): LSTMCell {
    return new LSTMCell(

      options.get(nodeName),
      options.get(LSTMCellOptions.InputSize),
      options.get(LSTMCellOptions.HiddenSize),
      options.get(LSTMCellOptions.Bias),
    );
  }

  public initCode(): string {
    return `LSTMCell(input_size=${this.InputSize}, hidden_size=${this.HiddenSize}, bias=${this.Bias})`;
  }
}
