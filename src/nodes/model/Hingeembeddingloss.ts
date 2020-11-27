import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum HingeEmbeddingLossOptions {
  Margin = 'Margin',
  SizeAverage = 'Size average',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class HingeEmbeddingLoss extends Node {
  type = ModelNodes.HingeEmbeddingLoss;
  name = ModelNodes.HingeEmbeddingLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(HingeEmbeddingLossOptions.Margin, TypeOptions.SliderOption, 1.0);
    this.addOption(HingeEmbeddingLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(HingeEmbeddingLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(HingeEmbeddingLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
