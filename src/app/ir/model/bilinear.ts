import { BilinearOptions } from '@/nodes/model/Bilinear';
import { nodeName } from '@/app/ir/irCommon';

export default class Bilinear {
  constructor(
  public readonly name: string,
  public readonly in1_features: bigint,
  public readonly in2_features: bigint,
  public readonly out_features: bigint,
  public readonly bias: boolean,
  ) {
  }

  static build(options: Map<string, any>): Bilinear {
    return new Bilinear(
      options.get(nodeName),
      options.get(BilinearOptions.In1Features),
      options.get(BilinearOptions.In2Features),
      options.get(BilinearOptions.OutFeatures),
      options.get(BilinearOptions.Bias),
    );
  }

  public initCode(): string {
    return `Bilinear(in1_features= ${this.in1_features}, in2_features= ${this.in2_features}, out_features= ${this.out_features}, bias= ${this.bias})`;
  }
}
