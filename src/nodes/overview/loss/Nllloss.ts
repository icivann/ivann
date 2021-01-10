import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum NLLLossOptions {
  Weight = 'Weight',
  SizeAverage = 'Size average',
  IgnoreIndex = 'Ignore index',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class NLLLoss extends Node {
  type = OverviewNodes.NLLLoss;
  name = OverviewNodes.NLLLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(NLLLossOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(NLLLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(NLLLossOptions.IgnoreIndex, TypeOptions.IntOption, -100);
    this.addOption(NLLLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(NLLLossOptions.Reduction, TypeOptions.DropdownOption, 'mean', undefined, { items: valuesOf(Reduction) });
  }
}
