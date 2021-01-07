import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum RMSpropOptions {
  Lr = 'Lr',
  Alpha = 'Alpha',
  Eps = 'Eps',
  WeightDecay = 'Weight decay',
  Momentum = 'Momentum',
  Centered = 'Centered'
}
export default class RMSprop extends Node {
  type = OverviewNodes.RMSprop;
  name = OverviewNodes.RMSprop;

  constructor() {
    super();
    this.addInputInterface('Model');
    this.addOutputInterface('Output');
    this.addOption(RMSpropOptions.Lr, TypeOptions.SliderOption, 0.01);
    this.addOption(RMSpropOptions.Alpha, TypeOptions.SliderOption, 0.99);
    this.addOption(RMSpropOptions.Eps, TypeOptions.SliderOption, 1e-08);
    this.addOption(RMSpropOptions.WeightDecay, TypeOptions.SliderOption, 0);
    this.addOption(RMSpropOptions.Momentum, TypeOptions.SliderOption, 0);
    this.addOption(RMSpropOptions.Centered, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
