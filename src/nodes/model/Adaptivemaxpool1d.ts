import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdaptiveMaxPool1dOptions {
  OutputSize = 'Output size',
  ReturnIndices = 'Return indices'
}
export default class AdaptiveMaxPool1d extends Node {
  type = ModelNodes.AdaptiveMaxPool1d;
  name = ModelNodes.AdaptiveMaxPool1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdaptiveMaxPool1dOptions.OutputSize, TypeOptions.VectorOption, [0]);
    this.addOption(AdaptiveMaxPool1dOptions.ReturnIndices, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
