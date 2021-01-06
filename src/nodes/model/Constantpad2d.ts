import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ConstantPad2dOptions {
  Padding = 'Padding',
  Value = 'Value'
}
export default class ConstantPad2d extends Node {
  type = ModelNodes.ConstantPad2d;
  name = ModelNodes.ConstantPad2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ConstantPad2dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0, 0]);
    this.addOption(ConstantPad2dOptions.Value, TypeOptions.SliderOption, 0.0);
  }
}
