import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdamWOptions {
  Lr = 'Lr',
  Betas = 'Betas',
  Eps = 'Eps',
  WeightDecay = 'Weight decay',
  Amsgrad = 'Amsgrad'
}
export default class AdamW extends Node {
  type = OverviewNodes.AdamW;
  name = OverviewNodes.AdamW;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdamWOptions.Lr, TypeOptions.SliderOption, 0.001);
    this.addOption(AdamWOptions.Betas, TypeOptions.VectorOption, [0.9, 0.999]);
    this.addOption(AdamWOptions.Eps, TypeOptions.SliderOption, 1e-08);
    this.addOption(AdamWOptions.WeightDecay, TypeOptions.SliderOption, 0.01);
    this.addOption(AdamWOptions.Amsgrad, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
