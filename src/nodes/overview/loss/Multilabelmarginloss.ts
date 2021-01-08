import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

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
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MultiLabelMarginLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(MultiLabelMarginLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(MultiLabelMarginLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
