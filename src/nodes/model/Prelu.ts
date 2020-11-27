import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum PReLUOptions {
  NumParameters = 'Num parameters',
  Init = 'Init'
}
export default class PReLU extends Node {
  type = ModelNodes.PReLU;
  name = ModelNodes.PReLU;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(PReLUOptions.NumParameters, TypeOptions.IntOption, 1);
    this.addOption(PReLUOptions.Init, TypeOptions.SliderOption, 0.25);
  }
}
