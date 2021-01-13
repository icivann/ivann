import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { Reduction } from '@/app/ir/irCommon';

export enum CosineEmbeddingLossOptions {
  Margin = 'Margin',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class CosineEmbeddingLoss extends Node {
  type = OverviewNodes.CosineEmbeddingLoss;
  name = OverviewNodes.CosineEmbeddingLoss;

  constructor() {
    super();

    this.addOutputInterface('Output');
    this.addOption(CosineEmbeddingLossOptions.Margin, TypeOptions.SliderOption, 0.0);
    this.addOption(CosineEmbeddingLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(CosineEmbeddingLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(CosineEmbeddingLossOptions.Reduction, TypeOptions.DropdownOption, 'mean',
      undefined, { items: valuesOf(Reduction) });
  }
}
