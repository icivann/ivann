import { ReplicationPad3dOptions } from '@/nodes/model/Replicationpad3d';
import { nodeName } from '@/app/ir/irCommon';

export default class ReplicationPad3d {
  constructor(
  public readonly name: string,
  public readonly Padding: [bigint, bigint, bigint, bigint, bigint, bigint],
  ) {
  }

  static build(options: Map<string, any>): ReplicationPad3d {
    return new ReplicationPad3d(

      options.get(nodeName),
      [options.get(ReplicationPad3dOptions.Padding)[0], options.get(ReplicationPad3dOptions.Padding)[1], options.get(ReplicationPad3dOptions.Padding)[2],
        options.get(ReplicationPad3dOptions.Padding)[3], options.get(ReplicationPad3dOptions.Padding)[4], options.get(ReplicationPad3dOptions.Padding)[5]],
    );
  }

  public initCode(): string {
    return `ReplicationPad3d(padding=(${this.Padding}))`;
  }
}
