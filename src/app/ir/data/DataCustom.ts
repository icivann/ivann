import parse from '@/app/parser/parser';
import { CustomOptions } from '@/nodes/common/Custom';
import { nodeName } from '@/app/ir/irCommon';
import { DataCustomOptions } from '@/nodes/data/DataCustom';

class DataCustom {
  constructor(
    public readonly name: string,
    public readonly code: string,
    public readonly dataLoading: boolean,
  ) {
  }

  static build(options: Map<string, any>): DataCustom {
    return new DataCustom(
      options.get(nodeName),
      options.get(CustomOptions.Code),
      options.get(DataCustomOptions.DATA_LOADING),
    );
  }

  public initCode(): string {
    return this.code;
  }

  public callCode(params: string[]): string[] {
    const parsedFuncs = parse(this.code);

    if (parsedFuncs instanceof Error) {
      console.error(parsedFuncs);
      return ['CUSTOM_NODE_PARSE_ERROR'];
    }
    if (parsedFuncs.length > 0) {
      const parsedFunc = parsedFuncs[0];
      return [`${parsedFunc.name}(${params})`];
    }
    return ['CUSTOM_NODE_NO_FUNCTION_ERROR'];
  }
}

export default DataCustom;
