import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ConstantPad1dOptions {
  Padding = 'Padding',
  Value = 'Value'
}
export default class ConstantPad1d extends Node {
  type = ModelNodes.ConstantPad1d;
  name = ModelNodes.ConstantPad1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ConstantPad1dOptions.Padding, TypeOptions.VectorOption, [0, 0]);
    this.addOption(ConstantPad1dOptions.Value, TypeOptions.SliderOption, 0.0);
  }
}
