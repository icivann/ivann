import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdamOptions {
  Lr = 'Lr',
  Betas = 'Betas',
  Eps = 'Eps',
  WeightDecay = 'Weight decay',
  Amsgrad = 'Amsgrad'
}
export default class Adam extends Node {
  type = OverviewNodes.Adam;
  name = OverviewNodes.Adam;

  constructor() {
    super();
    this.addInputInterface('Model');
    this.addOutputInterface('Output');
    this.addOption(AdamOptions.Lr, TypeOptions.SliderOption, 0.001);
    this.addOption(AdamOptions.Betas, TypeOptions.VectorOption, [0.9, 0.999]);
    this.addOption(AdamOptions.Eps, TypeOptions.SliderOption, 1e-08);
    this.addOption(AdamOptions.WeightDecay, TypeOptions.SliderOption, 0);
    this.addOption(AdamOptions.Amsgrad, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
