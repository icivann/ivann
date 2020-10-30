import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum Dropout3dOptions {
  P = 'P',
  Inplace = 'Inplace'
}
export default class Dropout3d extends Node {
  type = Nodes.Dropout3d;//TODO add layer type
  name = Nodes.Dropout3d;

constructor() {
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  this.addOption(Dropout3dOptions.P, TypeOptions.SliderOption, 0.5);
  this.addOption(Dropout3dOptions.Inplace, TypeOptions.TickBoxOption);
  }
}
