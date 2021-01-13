import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum MultiMarginLossOptions {
  P = 'P',
  Margin = 'Margin',
  Weight = 'Weight',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class MultiMarginLoss extends Node {
  type = OverviewNodes.MultiMarginLoss;
  name = OverviewNodes.MultiMarginLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(MultiMarginLossOptions.P, TypeOptions.IntOption, 1);
    this.addOption(MultiMarginLossOptions.Margin, TypeOptions.SliderOption, 1.0);
    this.addOption(MultiMarginLossOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(MultiMarginLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(MultiMarginLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(MultiMarginLossOptions.Reduction, TypeOptions.DropdownOption, 'mean', undefined, { items: valuesOf(Reduction) });
  }
}
