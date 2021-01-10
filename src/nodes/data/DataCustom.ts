import Custom from '@/nodes/common/Custom';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { DataNodes } from '@/nodes/data/Types';

export enum DataCustomOptions {
  DATA_LOADING = 'Data Loading',
}
export default class DataCustom extends Custom {
  type = DataNodes.DataCustom;

  constructor(parsedFunction?: ParsedFunction) {
    super(parsedFunction);
    this.addOption(DataCustomOptions.DATA_LOADING,
      TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
