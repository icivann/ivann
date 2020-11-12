import { nodeName } from '@/app/ir/irCommon';

class Concat {
  constructor(
    public readonly name: string,
  ) {}
  static build(options: Map<string, any>): Concat {
    return new Concat(options.get(nodeName));
  }
}

export default Concat;
