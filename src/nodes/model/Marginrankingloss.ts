import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum MarginRankingLossOptions {
  Margin = 'Margin',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class MarginRankingLoss extends Node {
  type = ModelNodes.MarginRankingLoss;
  name = ModelNodes.MarginRankingLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MarginRankingLossOptions.Margin, TypeOptions.SliderOption, 0.0);
    this.addOption(MarginRankingLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(MarginRankingLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(MarginRankingLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
