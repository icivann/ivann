import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum MSELossOptions {
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class MSELoss extends Node {
  type = ModelNodes.MSELoss;
  name = ModelNodes.MSELoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MSELossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(MSELossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(MSELossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
