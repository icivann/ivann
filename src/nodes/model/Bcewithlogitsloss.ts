import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum BCEWithLogitsLossOptions {
  Weight = 'Weight',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction',
  PosWeight = 'Pos weight'
}
export default class BCEWithLogitsLoss extends Node {
  type = ModelNodes.BCEWithLogitsLoss;
  name = ModelNodes.BCEWithLogitsLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(BCEWithLogitsLossOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(BCEWithLogitsLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(BCEWithLogitsLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(BCEWithLogitsLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
    this.addOption(BCEWithLogitsLossOptions.PosWeight, TypeOptions.VectorOption, 0);
  }
}
