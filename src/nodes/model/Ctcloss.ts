import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum CTCLossOptions {
  Blank = 'Blank',
  Reduction = 'Reduction',
  ZeroInfinity = 'Zero infinity'
}
export default class CTCLoss extends Node {
  type = ModelNodes.CTCLoss;
  name = ModelNodes.CTCLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(CTCLossOptions.Blank, TypeOptions.IntOption, 0);
    this.addOption(CTCLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
    this.addOption(CTCLossOptions.ZeroInfinity, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
