import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum SmoothL1LossOptions {
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction',
  Beta = 'Beta'
}
export default class SmoothL1Loss extends Node {
  type = OverviewNodes.SmoothL1Loss;
  name = OverviewNodes.SmoothL1Loss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(SmoothL1LossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(SmoothL1LossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(SmoothL1LossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
    this.addOption(SmoothL1LossOptions.Beta, TypeOptions.SliderOption, 1.0);
  }
}
