import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum CosineSimilarityOptions {
  Dim = 'Dim',
  Eps = 'Eps'
}
export default class CosineSimilarity extends Node {
  type = ModelNodes.CosineSimilarity;
  name = ModelNodes.CosineSimilarity;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(CosineSimilarityOptions.Dim, TypeOptions.IntOption, 0);
    this.addOption(CosineSimilarityOptions.Eps, TypeOptions.SliderOption, 0.0);
  }
}
