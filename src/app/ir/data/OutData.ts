import { nodeName } from '@/app/ir/irCommon';

class OutData {
  constructor(
    public readonly name: string,
    batchSize: BigInt,
  ) {
  }

  static build(options: Map<string, any>): OutData {
    return new OutData(
      options.get(nodeName),
      options.get('BatchSize'),
    );
  }
}

export default OutData;
