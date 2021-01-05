import { LogSoftmaxOptions } from '@/nodes/model/Logsoftmax';
import { nodeName } from '@/app/ir/irCommon';

export default class LogSoftmax {
  constructor(
  public readonly name: string,
  public readonly Dim: [bigint],
  ) {
  }

  static build(options: Map<string, any>): LogSoftmax {
    return new LogSoftmax(

      options.get(nodeName),
      [options.get(LogSoftmaxOptions.Dim)[0]],
    );
  }

  public initCode(): string {
    return `LogSoftmax(Dim= ${this.Dim})`;
  }
}
