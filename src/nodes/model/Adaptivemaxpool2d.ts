import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdaptiveMaxPool2dOptions {
  OutputSize = 'Output size',
  ReturnIndices = 'Return indices'
}
export default class AdaptiveMaxPool2d extends Node {
  type = ModelNodes.AdaptiveMaxPool2d;
  name = ModelNodes.AdaptiveMaxPool2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdaptiveMaxPool2dOptions.OutputSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(AdaptiveMaxPool2dOptions.ReturnIndices, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
