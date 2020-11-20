import { nodeName } from '@/app/ir/irCommon';

class LoadCsv {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): LoadCsv {
    return new LoadCsv(
      options.get(nodeName),
    );
  }
}

export default LoadCsv;
