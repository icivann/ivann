import { ReplicationPad2dOptions } from '@/nodes/model/Replicationpad2d';
import { nodeName } from '@/app/ir/irCommon';

export default class ReplicationPad2d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint, bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): ReplicationPad2d {
    return new ReplicationPad2d(

      options.get(nodeName),
      [options.get(ReplicationPad2dOptions.Padding)[0], options.get(ReplicationPad2dOptions.Padding)[1],
        options.get(ReplicationPad2dOptions.Padding)[2], options.get(ReplicationPad2dOptions.Padding)[3]],
    );
  }

  public initCode(): string {
    return `ReplicationPad2d(Padding=(${this.Padding}))`;
  }
}
