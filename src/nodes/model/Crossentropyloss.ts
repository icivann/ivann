import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum CrossEntropyLossOptions {
  Weight = 'Weight',
  SizeAverage = 'Size average',
  IgnoreIndex = 'Ignore index',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class CrossEntropyLoss extends Node {
  type = ModelNodes.CrossEntropyLoss;
  name = ModelNodes.CrossEntropyLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(CrossEntropyLossOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(CrossEntropyLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(CrossEntropyLossOptions.IgnoreIndex, TypeOptions.IntOption, -100);
    this.addOption(CrossEntropyLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(CrossEntropyLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
