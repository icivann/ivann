import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum MarginRankingLossOptions {
  Margin = 'Margin',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class MarginRankingLoss extends Node {
  type = OverviewNodes.MarginRankingLoss;
  name = OverviewNodes.MarginRankingLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(MarginRankingLossOptions.Margin, TypeOptions.SliderOption, 0.0);
    this.addOption(MarginRankingLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(MarginRankingLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(MarginRankingLossOptions.Reduction, TypeOptions.DropdownOption, 'mean', undefined, { items: valuesOf(Reduction) });
  }
}
