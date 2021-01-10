import { CustomOptions } from '@/nodes/common/Custom';
import { nodeName } from '@/app/ir/irCommon';
import { OverviewCustomOptions } from '@/nodes/overview/OverviewCustom';
import Custom from '@/app/ir/Custom';

class OverviewCustom extends Custom {
  public readonly trainer: boolean;

  constructor(name: string, code: string, trainer: boolean) {
    super(name, code);
    this.trainer = trainer;
  }

  static build(options: Map<string, any>): OverviewCustom {
    return new OverviewCustom(
      options.get(nodeName),
      options.get(CustomOptions.Code),
      options.get(OverviewCustomOptions.TRAINER),
    );
  }
}

export default OverviewCustom;
