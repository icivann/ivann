import { CustomOptions } from '@/nodes/common/Custom';
import { nodeName } from '@/app/ir/irCommon';
import { DataCustomOptions } from '@/nodes/data/DataCustom';
import Custom from '@/app/ir/Custom';
import parse, { getIndentation } from '@/app/parser/parser';

class DataCustom extends Custom {
  public readonly dataLoading: boolean;

  constructor(name: string, code: string, dataLoading: boolean) {
    super(name, code);
    this.dataLoading = dataLoading;
  }

  static build(options: Map<string, any>): DataCustom {
    return new DataCustom(
      options.get(nodeName),
      options.get(CustomOptions.Code),
      options.get(DataCustomOptions.DATA_LOADING),
    );
  }

  public initCode(): string[] {
    const func = parse(this.code);

    if (func instanceof Error) {
      return ['CUSTOM_NODE_PARSE_ERROR'];
    }

    let body = func[0].body.split('\n').splice(0).join('\n');
    const indent = getIndentation(body) as string;
    body = body.split('\n').map((line) => line.substring(indent.length)).join('\n');

    console.log('lol');
    console.log(body);

    const parsedFuncs = parse(body);

    if (parsedFuncs instanceof Error) {
      console.error(parsedFuncs);
      return ['CUSTOM_NODE_PARSE_ERROR'];
    }
    if (parsedFuncs.length > 0) {
      const parsedFunc = parsedFuncs[0];
      return parsedFunc.body.split('\n').map((line) => line.substring(indent.length));
    }
    return ['CUSTOM_NODE_NO_FUNCTION_ERROR'];
  }

  public callCode(): string[] {
    const func = parse(this.code);

    if (func instanceof Error) {
      return ['CUSTOM_NODE_PARSE_ERROR'];
    }

    let body = func[0].body.split('\n').splice(0).join('\n');
    const indent = getIndentation(body) as string;
    body = body.split('\n').map((line) => line.substring(indent.length)).join('\n');
    const parsedFuncs = parse(body);

    if (parsedFuncs instanceof Error) {
      console.error(parsedFuncs);
      return ['CUSTOM_NODE_PARSE_ERROR'];
    }
    if (parsedFuncs.length > 1) {
      const parsedFunc = parsedFuncs[1];
      return parsedFunc.body.split('\n').map((line) => line.substring(indent.length));
    }
    return ['CUSTOM_NODE_NO_FUNCTION_ERROR'];
  }
}

export default DataCustom;
