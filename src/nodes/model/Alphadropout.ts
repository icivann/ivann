import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AlphaDropoutOptions {
  P = 'P',
  Inplace = 'Inplace'
}
export default class AlphaDropout extends Node {
  type = ModelNodes.AlphaDropout;
  name = ModelNodes.AlphaDropout;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AlphaDropoutOptions.P, TypeOptions.SliderOption, 0.0);
    this.addOption(AlphaDropoutOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
