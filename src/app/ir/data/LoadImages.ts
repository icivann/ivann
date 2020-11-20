import { nodeName } from '@/app/ir/irCommon';

class LoadImages {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): LoadImages {
    return new LoadImages(
      options.get(nodeName),
    );
  }
}

export default LoadImages;
