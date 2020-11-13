import { nodeName } from '@/app/ir/irCommon';

class Model {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Model {
    return new Model(
      options.get(nodeName),
    );
  }
}

export default Model;
