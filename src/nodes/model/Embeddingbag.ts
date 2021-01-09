import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { Mode } from '@/app/ir/irCommon';

export enum EmbeddingBagOptions {
  NumEmbeddings = 'Num embeddings',
  EmbeddingDim = 'Embedding dim',
  MaxNorm = 'Max norm',
  NormType = 'Norm type',
  ScaleGradByFreq = 'Scale grad by freq',
  Mode = 'Mode',
  Sparse = 'Sparse',
  Weight = ' weight',
  IncludeLastOffset = 'Include last offset'
}
export default class EmbeddingBag extends Node {
  type = ModelNodes.EmbeddingBag;
  name = ModelNodes.EmbeddingBag;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(EmbeddingBagOptions.NumEmbeddings, TypeOptions.IntOption, 0);
    this.addOption(EmbeddingBagOptions.EmbeddingDim, TypeOptions.IntOption, 0);
    this.addOption(EmbeddingBagOptions.MaxNorm, TypeOptions.VectorOption, [0]);
    this.addOption(EmbeddingBagOptions.NormType, TypeOptions.SliderOption, 0.0);
    this.addOption(EmbeddingBagOptions.ScaleGradByFreq, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(EmbeddingBagOptions.Mode, TypeOptions.DropdownOption, 'mean', undefined, { items: { Mode } });
    this.addOption(EmbeddingBagOptions.Sparse, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(EmbeddingBagOptions.Weight, TypeOptions.VectorOption, [0]);
    this.addOption(EmbeddingBagOptions.IncludeLastOffset, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
