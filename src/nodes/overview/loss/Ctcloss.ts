import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

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

    this.addOutputInterface('Output');
    this.addOption(CTCLossOptions.Blank, TypeOptions.IntOption, 0);
    this.addOption(CTCLossOptions.Reduction, TypeOptions.DropdownOption, 'mean',
      undefined, { items: valuesOf(Reduction) });
    this.addOption(CTCLossOptions.ZeroInfinity, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
