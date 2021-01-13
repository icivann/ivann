import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum TripletMarginLossOptions {
  Margin = 'Margin',
  P = 'P',
  Eps = 'Eps',
  Swap = 'Swap',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class TripletMarginLoss extends Node {
  type = OverviewNodes.TripletMarginLoss;
  name = OverviewNodes.TripletMarginLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(TripletMarginLossOptions.Margin, TypeOptions.SliderOption, 1.0);
    this.addOption(TripletMarginLossOptions.P, TypeOptions.SliderOption, 2.0);
    this.addOption(TripletMarginLossOptions.Eps, TypeOptions.SliderOption, 1e-06);
    this.addOption(TripletMarginLossOptions.Swap, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(TripletMarginLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(TripletMarginLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(TripletMarginLossOptions.Reduction, TypeOptions.DropdownOption, 'mean', undefined, { items: valuesOf(Reduction) });
  }
}
