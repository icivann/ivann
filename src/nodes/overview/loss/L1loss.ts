import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum L1LossOptions {
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class L1Loss extends Node {
  type = OverviewNodes.L1Loss;
  name = OverviewNodes.L1Loss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(L1LossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(L1LossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(L1LossOptions.Reduction, TypeOptions.DropdownOption, 'mean', undefined, { items: valuesOf(Reduction) });
  }
}
