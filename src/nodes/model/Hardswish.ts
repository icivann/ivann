import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum HardswishOptions {
  Inplace = 'Inplace'
}
export default class Hardswish extends Node {
  type = ModelNodes.Hardswish;
  name = ModelNodes.Hardswish;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(HardswishOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
