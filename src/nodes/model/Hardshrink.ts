import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum HardshrinkOptions {
  Lambd = 'Lambd'
}
export default class Hardshrink extends Node {
  type = ModelNodes.Hardshrink;
  name = ModelNodes.Hardshrink;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(HardshrinkOptions.Lambd, TypeOptions.SliderOption, 0.5);
  }
}
