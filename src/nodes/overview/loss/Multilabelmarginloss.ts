import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum MultiLabelMarginLossOptions {
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class MultiLabelMarginLoss extends Node {
  type = OverviewNodes.MultiLabelMarginLoss;
  name = OverviewNodes.MultiLabelMarginLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(MultiLabelMarginLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(MultiLabelMarginLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(MultiLabelMarginLossOptions.Reduction, TypeOptions.DropdownOption, 'mean', undefined, { items: valuesOf(Reduction) });
  }
}
