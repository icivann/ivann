import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum KLDivLossOptions {
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction',
  LogTarget = 'Log target'
}
export default class KLDivLoss extends Node {
  type = ModelNodes.KLDivLoss;
  name = ModelNodes.KLDivLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(KLDivLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(KLDivLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(KLDivLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
    this.addOption(KLDivLossOptions.LogTarget, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
