import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LeakyReLUOptions {
  NegativeSlope = 'Negative slope',
  Inplace = 'Inplace'
}
export default class LeakyReLU extends Node {
  type = ModelNodes.LeakyReLU;
  name = ModelNodes.LeakyReLU;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(LeakyReLUOptions.NegativeSlope, TypeOptions.SliderOption, 0.01);
    this.addOption(LeakyReLUOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
