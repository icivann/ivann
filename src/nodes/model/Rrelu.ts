import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum RReLUOptions {
  Lower = 'Lower',
  Upper = 'Upper',
  Inplace = 'Inplace'
}
export default class RReLU extends Node {
  type = ModelNodes.RReLU;
  name = ModelNodes.RReLU;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(RReLUOptions.Lower, TypeOptions.SliderOption, 0.125);
    this.addOption(RReLUOptions.Upper, TypeOptions.SliderOption, 0.3333333333333333);
    this.addOption(RReLUOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
