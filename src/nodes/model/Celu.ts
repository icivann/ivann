import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum CELUOptions {
  Alpha = 'Alpha',
  Inplace = 'Inplace'
}
export default class CELU extends Node {
  type = ModelNodes.CELU;
  name = ModelNodes.CELU;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(CELUOptions.Alpha, TypeOptions.SliderOption, 1.0);
    this.addOption(CELUOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
