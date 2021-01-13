import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum MultiLabelSoftMarginLossOptions {
  Weight = 'Weight',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class MultiLabelSoftMarginLoss extends Node {
  type = OverviewNodes.MultiLabelSoftMarginLoss;
  name = OverviewNodes.MultiLabelSoftMarginLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(MultiLabelSoftMarginLossOptions.Weight, TypeOptions.VectorOption, 0);
    this.addOption(MultiLabelSoftMarginLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(MultiLabelSoftMarginLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(MultiLabelSoftMarginLossOptions.Reduction, TypeOptions.DropdownOption, 'mean', undefined, { items: valuesOf(Reduction) });
  }
}
