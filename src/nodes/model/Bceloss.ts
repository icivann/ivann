import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum BCELossOptions {
  Weight = 'Weight',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class BCELoss extends Node {
  type = ModelNodes.BCELoss;
  name = ModelNodes.BCELoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(BCELossOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(BCELossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(BCELossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(BCELossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
