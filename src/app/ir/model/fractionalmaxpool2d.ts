import { FractionalMaxPool2dOptions } from '@/nodes/model/Fractionalmaxpool2d';
import { nodeName } from '@/app/ir/irCommon';

export default class FractionalMaxPool2d {
  constructor(
  public readonly name: string,
  public readonly KernelSize: [bigint, bigint],
  public readonly OutputSize: [bigint, bigint],
  public readonly OutputRatio: [bigint, bigint],
  public readonly ReturnIndices: boolean,
  ) {
  }

  static build(options: Map<string, any>): FractionalMaxPool2d {
    return new FractionalMaxPool2d(

      options.get(nodeName),
      [options.get(FractionalMaxPool2dOptions.KernelSize)[0], options.get(FractionalMaxPool2dOptions.KernelSize)[1]],
      [options.get(FractionalMaxPool2dOptions.OutputSize)[0], options.get(FractionalMaxPool2dOptions.OutputSize)[1]],
      [options.get(FractionalMaxPool2dOptions.OutputRatio)[0], options.get(FractionalMaxPool2dOptions.OutputRatio)[1]],
      options.get(FractionalMaxPool2dOptions.ReturnIndices),
    );
  }

  public initCode(): string {
    return `FractionalMaxPool2d(KernelSize=, (${this.KernelSize}), OutputSize=, (${this.OutputSize}), OutputRatio=, (${this.OutputRatio}), ReturnIndices=, (${this.ReturnIndices}))`;
  }
}
