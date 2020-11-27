import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum NLLLossOptions {
  Weight = 'Weight',
  SizeAverage = 'Size average',
  IgnoreIndex = 'Ignore index',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class NLLLoss extends Node {
  type = ModelNodes.NLLLoss;
  name = ModelNodes.NLLLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(NLLLossOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(NLLLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(NLLLossOptions.IgnoreIndex, TypeOptions.IntOption, -100);
    this.addOption(NLLLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(NLLLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
