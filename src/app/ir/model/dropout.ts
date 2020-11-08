import { DropoutOptions } from '@/nodes/model/DropoutBaklava';

export default class Dropout {
  constructor(
  public readonly p: number,
  public readonly inplace: boolean,
  ) {
  }

  static build(options: Map<string, any>): Dropout {
    return new Dropout(
      options.get(DropoutOptions.P),
      options.get(DropoutOptions.Inplace),
    );
  }

  public initCode(): string {
    return `Dropout(p=${this.p}, inplace=${this.inplace})`;
  }
}
