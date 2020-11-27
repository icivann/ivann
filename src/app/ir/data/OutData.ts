import { nodeName } from '@/app/ir/irCommon';

class OutData {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): OutData {
    return new OutData(
      options.get(nodeName),
    );
  }
}

export default OutData;
