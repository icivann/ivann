import { ReplicationPad1dOptions } from '@/nodes/model/Replicationpad1d';
import { nodeName } from '@/app/ir/irCommon';

export default class ReplicationPad1d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): ReplicationPad1d {
    return new ReplicationPad1d(

      options.get(nodeName),
      [options.get(ReplicationPad1dOptions.Padding)[0], options.get(ReplicationPad1dOptions.Padding)[1]],
    );
  }

  public initCode(): string {
    return `ReplicationPad1d(Padding=, ${this.Padding})`;
  }
}
