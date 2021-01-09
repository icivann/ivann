import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ASGDOptions {
  Lr = 'Lr',
  Lambd = 'Lambd',
  Alpha = 'Alpha',
  T0 = 'T0',
  WeightDecay = 'Weight decay'
}
export default class ASGD extends Node {
  type = OverviewNodes.ASGD;
  name = OverviewNodes.ASGD;

  constructor() {
    super();
    this.addInputInterface('Model');
    this.addOutputInterface('Output');
    this.addOption(ASGDOptions.Lr, TypeOptions.SliderOption, 0.01);
    this.addOption(ASGDOptions.Lambd, TypeOptions.SliderOption, 0.0001);
    this.addOption(ASGDOptions.Alpha, TypeOptions.SliderOption, 0.75);
    this.addOption(ASGDOptions.T0, TypeOptions.SliderOption, 1000000.0);
    this.addOption(ASGDOptions.WeightDecay, TypeOptions.SliderOption, 0);
  }
}
