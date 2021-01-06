import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ZeroPad2dOptions {
  Padding = 'Padding'
}
export default class ZeroPad2d extends Node {
  type = ModelNodes.ZeroPad2d;
  name = ModelNodes.ZeroPad2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ZeroPad2dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0, 0]);
  }
}
