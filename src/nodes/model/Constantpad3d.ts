import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ConstantPad3dOptions {
  Padding = 'Padding',
  Value = 'Value'
}
export default class ConstantPad3d extends Node {
  type = ModelNodes.ConstantPad3d;
  name = ModelNodes.ConstantPad3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ConstantPad3dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0, 0, 0, 0]);
    this.addOption(ConstantPad3dOptions.Value, TypeOptions.SliderOption, 0.0);
  }
}
