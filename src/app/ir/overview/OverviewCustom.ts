import parse from '@/app/parser/parser';
import { CustomOptions } from '@/nodes/common/Custom';
import { nodeName } from '@/app/ir/irCommon';
import { OverviewCustomOptions } from '@/nodes/overview/OverviewCustom';

class OverviewCustom {
  constructor(
    public readonly name: string,
    public readonly code: string,
    public readonly trainer: boolean,
  ) {
  }

  static build(options: Map<string, any>): OverviewCustom {
    return new OverviewCustom(
      options.get(nodeName),
      options.get(CustomOptions.Code),
      options.get(OverviewCustomOptions.TRAINER),
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

export default OverviewCustom;
