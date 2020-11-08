import { Dropout2dOptions } from '@/nodes/model/Dropout2dBaklava';

export default class Dropout2d {
  constructor(
  public readonly p: number,
  public readonly inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): Dropout2d {
    return new Dropout2d(
      options.get(Dropout2dOptions.P),
      options.get(Dropout2dOptions.Inplace),
    );
  }

  public initCode(): string {
    return `Dropout2d(p=${this.p}, inplace=${this.inplace})`;
  }

  public callCode(params: string[], name: string): string {
    return `${name}(${params.join(', ')})`;
  }
}
