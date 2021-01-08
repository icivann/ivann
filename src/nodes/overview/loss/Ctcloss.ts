import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum CTCLossOptions {
  Blank = 'Blank',
  Reduction = 'Reduction',
  ZeroInfinity = 'Zero infinity'
}
export default class CTCLoss extends Node {
  type = OverviewNodes.CTCLoss;
  name = OverviewNodes.CTCLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(CTCLossOptions.Blank, TypeOptions.IntOption, 0);
    this.addOption(CTCLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
    this.addOption(CTCLossOptions.ZeroInfinity, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
