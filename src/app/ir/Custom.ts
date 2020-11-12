import { CustomOptions } from '@/nodes/model/custom/Custom';
import parse from '@/app/parser/parser';

class Custom {
  public readonly name = 'custom'
  constructor(
    public readonly code: string,
  ) {
  }

  static build(options: Map<string, any>): Custom {
    // TODO CORE-58 Change InlineCode Option to use state.parsedFunction
    return new Custom(
      options.get(CustomOptions.InlineCode).text,
    );
  }

  public callCode(params: string[], name: string): string {
    const parsedFuncs = parse(this.code);

    if (parsedFuncs instanceof Error) {
      console.error(parsedFuncs);
      return 'CUSTOM_NODE_PARSE_ERROR';
    } if (parsedFuncs.length > 0) {
      const parsedFunc = parsedFuncs[0];
      return `${parsedFunc.name}(${params})`;
    }
    return 'CUSTOM_NODE_NO_FUNCTION_ERROR';
  }
}

export default Custom;
