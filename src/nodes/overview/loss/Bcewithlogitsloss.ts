import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum BCEWithLogitsLossOptions {
  Weight = 'Weight',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction',
  PosWeight = 'Pos weight'
}
export default class BCEWithLogitsLoss extends Node {
  type = OverviewNodes.BCEWithLogitsLoss;
  name = OverviewNodes.BCEWithLogitsLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(BCEWithLogitsLossOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(BCEWithLogitsLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(BCEWithLogitsLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(BCEWithLogitsLossOptions.Reduction, TypeOptions.DropdownOption, 'mean',
      undefined, { items: valuesOf(Reduction) });
    this.addOption(BCEWithLogitsLossOptions.PosWeight, TypeOptions.VectorOption, 0);
  }
}
