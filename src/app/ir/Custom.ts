import parse from '@/app/parser/parser';

abstract class Custom {
  protected constructor(
    public readonly name: string,
    public readonly code: string,
  ) {
  }

  public initCode(): string[] {
    return [this.code];
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

export default Custom;
