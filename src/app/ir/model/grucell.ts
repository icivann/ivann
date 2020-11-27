import { GRUCellOptions } from '@/nodes/model/Grucell';
import { nodeName } from '@/app/ir/irCommon';

export default class GRUCell {
  constructor(
  public readonly name: string,
  public readonly InputSize: bigint,
  public readonly HiddenSize: bigint,
  public readonly Bias: boolean,
  ) {
  }

  static build(options: Map<string, any>): GRUCell {
    return new GRUCell(

      options.get(nodeName),
      options.get(GRUCellOptions.InputSize),
      options.get(GRUCellOptions.HiddenSize),
      options.get(GRUCellOptions.Bias),
    );
  }

  public initCode(): string {
    return `GRUCell(InputSize=, ${this.InputSize}, HiddenSize=, ${this.HiddenSize}, Bias=, ${this.Bias})`;
  }
}
