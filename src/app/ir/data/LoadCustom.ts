import { nodeName } from '@/app/ir/irCommon';

class LoadCustom {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): LoadCustom {
    return new LoadCustom(
      options.get(nodeName),
    );
  }
}

export default LoadCustom;
