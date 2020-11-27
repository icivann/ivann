import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum EmbeddingOptions {
  NumEmbeddings = 'Num embeddings',
  EmbeddingDim = 'Embedding dim',
  PaddingIdx = 'Padding idx',
  MaxNorm = 'Max norm',
  NormType = 'Norm type',
  ScaleGradByFreq = 'Scale grad by freq',
  Sparse = 'Sparse',
  Weight = ' weight'
}
export default class Embedding extends Node {
  type = ModelNodes.Embedding;
  name = ModelNodes.Embedding;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(EmbeddingOptions.NumEmbeddings, TypeOptions.IntOption, 0);
    this.addOption(EmbeddingOptions.EmbeddingDim, TypeOptions.IntOption, 0);
    this.addOption(EmbeddingOptions.PaddingIdx, TypeOptions.VectorOption, [0]);
    this.addOption(EmbeddingOptions.MaxNorm, TypeOptions.VectorOption, [0]);
    this.addOption(EmbeddingOptions.NormType, TypeOptions.SliderOption, 2.0);
    this.addOption(EmbeddingOptions.ScaleGradByFreq, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(EmbeddingOptions.Sparse, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(EmbeddingOptions.Weight, TypeOptions.VectorOption, 0);
  }
}
