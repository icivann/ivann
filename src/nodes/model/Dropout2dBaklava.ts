import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum Dropout2dOptions {
  P = 'P',
  Inplace = 'Inplace'
}
export default class Dropout2d extends Node {
  type = Nodes.Dropout2d;
  name = Nodes.Dropout2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(Dropout2dOptions.P, TypeOptions.SliderOption, 0.5);
    this.addOption(Dropout2dOptions.Inplace, TypeOptions.TickBoxOption);
  }
}
