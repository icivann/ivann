import { Softmax2dOptions } from '@/nodes/model/Softmax2d';
import { nodeName } from '@/app/ir/irCommon';

export default class Softmax2d {
  constructor(
  public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Softmax2d {
    return new Softmax2d(

      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'Softmax2d()';
  }
}
