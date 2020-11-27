import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum HardsigmoidOptions {
  Inplace = 'Inplace'
}
export default class Hardsigmoid extends Node {
  type = ModelNodes.Hardsigmoid;
  name = ModelNodes.Hardsigmoid;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(HardsigmoidOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
