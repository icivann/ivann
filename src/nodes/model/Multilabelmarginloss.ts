import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum MultiLabelMarginLossOptions {
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class MultiLabelMarginLoss extends Node {
  type = ModelNodes.MultiLabelMarginLoss;
  name = ModelNodes.MultiLabelMarginLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MultiLabelMarginLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(MultiLabelMarginLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(MultiLabelMarginLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
