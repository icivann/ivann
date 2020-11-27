import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ReflectionPad2dOptions {
  Padding = 'Padding'
}
export default class ReflectionPad2d extends Node {
  type = ModelNodes.ReflectionPad2d;
  name = ModelNodes.ReflectionPad2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ReflectionPad2dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0, 0]);
  }
}
