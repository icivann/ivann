import { LogSigmoidOptions } from '@/nodes/model/Logsigmoid';
import { nodeName } from '@/app/ir/irCommon';

export default class LogSigmoid {
  constructor(
  public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): LogSigmoid {
    return new LogSigmoid(

      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'LogSigmoid()';
  }
}
