import { nodeName } from '@/app/ir/irCommon';

class InData {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): InData {
    return new InData(
      options.get(nodeName),
    );
  }
}

export default InData;
