import { nodeName } from '@/app/ir/irCommon';
import { LoadCsvOptions } from '@/nodes/data/LoadCsv';

class LoadCsv {
  constructor(
    public readonly name: string,
    public readonly column: BigInt,
  ) {
  }

  static build(options: Map<string, any>): LoadCsv {
    return new LoadCsv(
      options.get(nodeName),
      options.get(LoadCsvOptions.Column),
    );
  }

  public initCode(name: string): string[] {
    return [
      `self.${name} = pd.read_csv(${name}_path)`,
      `self.${name} = np.asarray(self.${name}[:, ${this.column}])`,
    ];
  }

  public callCode(name: string): string[] {
    return [];
  }
}

export default LoadCsv;
