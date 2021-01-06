import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdaptiveLogSoftmaxWithLossOptions {
  InFeatures = 'In features',
  NClasses = 'N classes',
  DivValue = 'Div value',
  HeadBias = 'Head bias'
}
export default class AdaptiveLogSoftmaxWithLoss extends Node {
  type = ModelNodes.AdaptiveLogSoftmaxWithLoss;
  name = ModelNodes.AdaptiveLogSoftmaxWithLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdaptiveLogSoftmaxWithLossOptions.InFeatures, TypeOptions.IntOption, 0);
    this.addOption(AdaptiveLogSoftmaxWithLossOptions.NClasses, TypeOptions.IntOption, 0);
    this.addOption(AdaptiveLogSoftmaxWithLossOptions.DivValue, TypeOptions.SliderOption, 4.0);
    this.addOption(AdaptiveLogSoftmaxWithLossOptions.HeadBias, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
