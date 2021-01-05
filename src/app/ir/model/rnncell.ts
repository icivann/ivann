import { RNNCellOptions } from '@/nodes/model/Rnncell';
import { nodeName, Nonlinearity, getNonlinearity } from '@/app/ir/irCommon';

export default class RNNCell {
  constructor(
  public readonly name: string,
  public readonly InputSize: bigint,
  public readonly HiddenSize: bigint,
  public readonly Bias: boolean,
  public readonly Nonlinearity: Nonlinearity,
  ) {
  }

  static build(options: Map<string, any>): RNNCell {
    return new RNNCell(

      options.get(nodeName),
      options.get(RNNCellOptions.InputSize),
      options.get(RNNCellOptions.HiddenSize),
      options.get(RNNCellOptions.Bias),
      getNonlinearity(options.get(RNNCellOptions.Nonlinearity)),
    );
  }

  public initCode(): string {
    return `RNNCell(InputSize= ${this.InputSize}, HiddenSize= ${this.HiddenSize}, Bias= ${this.Bias}, Nonlinearity= ${this.Nonlinearity})`;
  }
}
