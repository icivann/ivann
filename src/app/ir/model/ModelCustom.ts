import { CustomOptions } from '@/nodes/common/Custom';
import { nodeName } from '@/app/ir/irCommon';
import Custom from '@/app/ir/Custom';

class ModelCustom extends Custom {
  static build(options: Map<string, any>): ModelCustom {
    return new ModelCustom(
      options.get(nodeName),
      options.get(CustomOptions.Code),
    );
  }
}

export default ModelCustom;
