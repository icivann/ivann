import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ELUOptions {
  Alpha = 'Alpha',
  Inplace = 'Inplace'
}
export default class ELU extends Node {
  type = ModelNodes.ELU;
  name = ModelNodes.ELU;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ELUOptions.Alpha, TypeOptions.SliderOption, 1.0);
    this.addOption(ELUOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
