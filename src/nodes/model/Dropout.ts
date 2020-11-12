import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum DropoutOptions {
  P = 'P',
  Inplace = 'Inplace'
}
export default class Dropout extends Node {
  type = ModelNodes.Dropout;
  name = ModelNodes.Dropout;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(DropoutOptions.P, TypeOptions.SliderOption, 0.5);
    this.addOption(DropoutOptions.Inplace, TypeOptions.TickBoxOption);
  }
}
