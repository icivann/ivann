import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { PaddingMode, Reduction } from '@/app/ir/irCommon';

export enum BCELossOptions {
  Weight = 'Weight',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class BCELoss extends Node {
  type = OverviewNodes.BCELoss;
  name = OverviewNodes.BCELoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(BCELossOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(BCELossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(BCELossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(BCELossOptions.Reduction, TypeOptions.DropdownOption, 'mean',
      undefined, { items: valuesOf(Reduction) });
  }
}
