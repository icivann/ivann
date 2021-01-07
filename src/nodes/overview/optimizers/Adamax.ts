import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AdamaxOptions {
  Lr = 'Lr',
  Betas = 'Betas',
  Eps = 'Eps',
  WeightDecay = 'Weight decay'
}
export default class Adamax extends Node {
  type = OverviewNodes.Adamax;
  name = OverviewNodes.Adamax;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AdamaxOptions.Lr, TypeOptions.SliderOption, 0.002);
    this.addOption(AdamaxOptions.Betas, TypeOptions.VectorOption, [0.9, 0.999]);
    this.addOption(AdamaxOptions.Eps, TypeOptions.SliderOption, 1e-08);
    this.addOption(AdamaxOptions.WeightDecay, TypeOptions.SliderOption, 0);
  }
}
