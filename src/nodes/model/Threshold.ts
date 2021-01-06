import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ThresholdOptions {
  Threshold = 'Threshold',
  Value = 'Value',
  Inplace = 'Inplace'
}
export default class Threshold extends Node {
  type = ModelNodes.Threshold;
  name = ModelNodes.Threshold;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ThresholdOptions.Threshold, TypeOptions.SliderOption, 0);
    this.addOption(ThresholdOptions.Value, TypeOptions.SliderOption, 0);
    this.addOption(ThresholdOptions.Inplace, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
