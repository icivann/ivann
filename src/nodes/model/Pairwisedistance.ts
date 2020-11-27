import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum PairwiseDistanceOptions {
  P = 'P',
  Eps = 'Eps',
  Keepdim = 'Keepdim'
}
export default class PairwiseDistance extends Node {
  type = ModelNodes.PairwiseDistance;
  name = ModelNodes.PairwiseDistance;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(PairwiseDistanceOptions.P, TypeOptions.SliderOption, 0.0);
    this.addOption(PairwiseDistanceOptions.Eps, TypeOptions.SliderOption, 0.0);
    this.addOption(PairwiseDistanceOptions.Keepdim, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
