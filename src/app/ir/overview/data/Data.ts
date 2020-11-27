import { nodeName } from '@/app/ir/irCommon';

class Data {
  constructor(
    public readonly name: string,
    public readonly paths: string[],
    public readonly batchSize: BigInt,
  ) {
  }

  static build(options: Map<string, any>): Data {
    const paths: string[] = [];

    options.forEach((v, k) => {
      if (k.endsWith('path')) {
        if (v.text === undefined) {
          paths.push('');
        } else {
          paths.push(v.text);
        }
      }
    });

    return new Data(
      options.get(nodeName),
      paths,
      options.get('BatchSize'),
    );
  }

  public initCode(params: string[]): string[] {
    return [
      `DataLoader(${this.name}(${this.paths.map((p) => `'${p}'`).join(', ')}), batch_size=${this.batchSize}, shuffle=True)`,
    ];
  }
}

export default Data;
