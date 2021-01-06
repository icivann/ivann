import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum CosineEmbeddingLossOptions {
  Margin = 'Margin',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class CosineEmbeddingLoss extends Node {
  type = ModelNodes.CosineEmbeddingLoss;
  name = ModelNodes.CosineEmbeddingLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(CosineEmbeddingLossOptions.Margin, TypeOptions.SliderOption, 0.0);
    this.addOption(CosineEmbeddingLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(CosineEmbeddingLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(CosineEmbeddingLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
