import { nodeName } from '@/app/ir/irCommon';

export default class Flatten {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Flatten {
    return new Flatten(
      options.get(nodeName),
    );
  }

  public callCode(params: string[], name: string): string {
    return `torch.flatten(${params.join(', ')})`;
  }
}
