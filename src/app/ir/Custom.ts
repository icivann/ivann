import parse from '@/app/parser/parser';
import { CustomOptions } from '@/nodes/common/Custom';
import { nodeName } from '@/app/ir/irCommon';

class Custom {
  constructor(
    public readonly name: string,
    public readonly code: string,
    public readonly file: string,
  ) {
  }

  static build(options: Map<string, any>): Custom {
    return new Custom(
      options.get(nodeName),
      options.get(CustomOptions.Code),
      options.get(CustomOptions.File),
    );
  }

  public callCode(params: string[], name: string): string {
    const parsedFuncs = parse(this.code);

    if (parsedFuncs instanceof Error) {
      console.error(parsedFuncs);
      return 'CUSTOM_NODE_PARSE_ERROR';
    }
    if (parsedFuncs.length > 0) {
      const parsedFunc = parsedFuncs[0];
      return `${parsedFunc.name}(${params})`;
    }
    return 'CUSTOM_NODE_NO_FUNCTION_ERROR';
  }
}

export default Custom;
