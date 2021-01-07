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
    return `LSTMCell(InputSize=${this.InputSize}, HiddenSize=${this.HiddenSize}, Bias=${this.Bias})`;
  }
}
