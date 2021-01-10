import { Node } from '@baklavajs/core';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { OverviewNodes } from '@/nodes/overview/Types';

export enum AdadeltaOptions {
  Lr = 'Lr',
  Rho = 'Rho',
  Eps = 'Eps',
  WeightDecay = 'Weight decay'
}
export default class Adadelta extends Node {
  type = OverviewNodes.Adadelta;
  name = OverviewNodes.Adadelta;

  constructor() {
    super();
    this.addInputInterface('Model');
    this.addOutputInterface('Output');
    this.addOption(AdadeltaOptions.Lr, TypeOptions.SliderOption, 1.0);
    this.addOption(AdadeltaOptions.Rho, TypeOptions.SliderOption, 0.9);
    this.addOption(AdadeltaOptions.Eps, TypeOptions.SliderOption, 1e-06);
    this.addOption(AdadeltaOptions.WeightDecay, TypeOptions.SliderOption, 0);
  }
}
