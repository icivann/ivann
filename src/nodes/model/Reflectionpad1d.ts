import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ReflectionPad1dOptions {
  Padding = 'Padding'
}
export default class ReflectionPad1d extends Node {
  type = ModelNodes.ReflectionPad1d;
  name = ModelNodes.ReflectionPad1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ReflectionPad1dOptions.Padding, TypeOptions.VectorOption, [0, 0]);
  }
}
