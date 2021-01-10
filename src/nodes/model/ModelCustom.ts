import Custom from '@/nodes/common/Custom';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { ModelNodes } from '@/nodes/model/Types';

export default class ModelCustom extends Custom {
  type = ModelNodes.ModelCustom;

  constructor(parsedFunction?: ParsedFunction) {
    super(parsedFunction);
  }
}
