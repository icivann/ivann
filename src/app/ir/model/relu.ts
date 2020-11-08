import { ReLUOptions } from '@/nodes/model/ReluBaklava';

export default class ReLU {
  constructor(
  public readonly inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): ReLU {
    return new ReLU(
      options.get(ReLUOptions.Inplace),
    );
  }

  public initCode(): string {
    return `ReLU(inplace=${this.inplace})`;
  }

  public callCode(params: string[], name: string): string {
    return `${name}(${params.join(', ')})`;
  }
}
