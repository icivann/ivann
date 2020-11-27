import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ReLU6Options {
  Inplace = 'Inplace'
}
export default class ReLU6 extends Node {
  type = ModelNodes.ReLU6;
  name = ModelNodes.ReLU6;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ReLU6Options.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
