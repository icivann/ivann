import { Node } from '@baklavajs/core';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { OverviewNodes } from '@/nodes/overview/Types';

export enum RpropOptions {
  Lr = 'Lr',
  Etas = 'Etas',
  StepSizes = 'Step sizes'
}
export default class Rprop extends Node {
  type = OverviewNodes.Rprop;
  name = OverviewNodes.Rprop;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(RpropOptions.Lr, TypeOptions.SliderOption, 0.01);
    this.addOption(RpropOptions.Etas, TypeOptions.VectorOption, [0.5, 1.2]);
    this.addOption(RpropOptions.StepSizes, TypeOptions.VectorOption, [1e-06, 50]);
  }
}
