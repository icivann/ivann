import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdagradOptions {
  Lr = 'Lr',
  LrDecay = 'Lr decay',
  WeightDecay = 'Weight decay',
  InitialAccumulatorValue = 'Initial accumulator value',
  Eps = 'Eps'
}
export default class Adagrad extends Node {
  type = OverviewNodes.Adagrad;
  name = OverviewNodes.Adagrad;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdagradOptions.Lr, TypeOptions.SliderOption, 0.001);
    this.addOption(AdagradOptions.LrDecay, TypeOptions.SliderOption, 0);
    this.addOption(AdagradOptions.WeightDecay, TypeOptions.SliderOption, 0);
    this.addOption(AdagradOptions.InitialAccumulatorValue, TypeOptions.SliderOption, 0);
    this.addOption(AdagradOptions.Eps, TypeOptions.SliderOption, 1e-10);
  }
}
