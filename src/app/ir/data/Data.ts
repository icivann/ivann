import { nodeName } from '@/app/ir/irCommon';

class Data {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Data {
    return new Data(
      options.get(nodeName),
    );
  }
}

export default Data;
