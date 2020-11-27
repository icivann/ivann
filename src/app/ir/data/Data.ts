import { nodeName } from '@/app/ir/irCommon';
import { DataOptions } from '@/nodes/overview/Data';

class Data {
  constructor(
    public readonly name: string,
    public readonly paths: string[],
    public readonly batchSize: BigInt,
  ) {
  }

  static build(options: Map<string, any>): Data {
    const paths: string[] = [];

    options.forEach((v, k) => {
      if (k.endsWith('path')) {
        if (v.text === undefined) {
          paths.push('');
        } else {
          paths.push(v.text);
        }
      }
    });

    return new Data(
      options.get(nodeName),
      paths,
      options.get(DataOptions.BatchSize),
    );
  }
}

export default Data;
