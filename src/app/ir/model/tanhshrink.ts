import { TanhshrinkOptions } from '@/nodes/model/Tanhshrink';
import { nodeName } from '@/app/ir/irCommon';

export default class Tanhshrink {
  constructor(
  public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Tanhshrink {
    return new Tanhshrink(

      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'Tanhshrink()';
  }
}
