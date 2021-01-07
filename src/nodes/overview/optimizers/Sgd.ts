import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum SGDOptions {
  Lr = 'Lr',
  Momentum = 'Momentum',
  Dampening = 'Dampening',
  WeightDecay = 'Weight decay',
  Nesterov = 'Nesterov'
}
export default class SGD extends Node {
  type = OverviewNodes.SGD;
  name = OverviewNodes.SGD;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(SGDOptions.Lr, TypeOptions.SliderOption, 0);
    this.addOption(SGDOptions.Momentum, TypeOptions.SliderOption, 0);
    this.addOption(SGDOptions.Dampening, TypeOptions.SliderOption, 0);
    this.addOption(SGDOptions.WeightDecay, TypeOptions.SliderOption, 0);
    this.addOption(SGDOptions.Nesterov, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
