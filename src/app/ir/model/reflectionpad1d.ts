import { ReflectionPad1dOptions } from '@/nodes/model/Reflectionpad1d';
import { nodeName } from '@/app/ir/irCommon';

export default class ReflectionPad1d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): ReflectionPad1d {
    return new ReflectionPad1d(

      options.get(nodeName),
      [options.get(ReflectionPad1dOptions.Padding)[0], options.get(ReflectionPad1dOptions.Padding)[1]],
    );
  }

  public initCode(): string {
    return `ReflectionPad1d(Padding=${this.Padding})`;
  }
}
