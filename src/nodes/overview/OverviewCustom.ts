import Custom from '@/nodes/common/Custom';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { OverviewNodes } from '@/nodes/overview/Types';

export enum OverviewCustomOptions {
  TRAINER = 'Trainer',
}
export default class OverviewCustom extends Custom {
  type = OverviewNodes.OverviewCustom;

  constructor(parsedFunction?: ParsedFunction) {
    super(parsedFunction);
    this.addOption(OverviewCustomOptions.TRAINER,
      TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
