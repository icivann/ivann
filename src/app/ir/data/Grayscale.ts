import { nodeName } from '@/app/ir/irCommon';

class Grayscale {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Grayscale {
    return new Grayscale(
      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'transforms.Grayscale()';
  }
}

export default Grayscale;
