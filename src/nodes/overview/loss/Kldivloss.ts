import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum KLDivLossOptions {
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction',
  LogTarget = 'Log target'
}
export default class KLDivLoss extends Node {
  type = OverviewNodes.KLDivLoss;
  name = OverviewNodes.KLDivLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(KLDivLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(KLDivLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(KLDivLossOptions.Reduction, TypeOptions.DropdownOption, 'mean',
      undefined, { items: valuesOf(Reduction) });
    this.addOption(KLDivLossOptions.LogTarget, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
