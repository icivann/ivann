import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum DropoutOptions {
  P = 'P',
  Inplace = 'Inplace'
}
export default class Dropout extends Node {
  type = Nodes.Dropout;
  name = Nodes.Dropout;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(DropoutOptions.P, TypeOptions.SliderOption, 0.5);
    this.addOption(DropoutOptions.Inplace, TypeOptions.TickBoxOption);
  }
}
