import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum SparseAdamOptions {
  Lr = 'Lr',
  Betas = 'Betas',
  Eps = 'Eps'
}
export default class SparseAdam extends Node {
  type = OverviewNodes.SparseAdam;
  name = OverviewNodes.SparseAdam;

  constructor() {
    super();
    this.addInputInterface('Model');
    this.addOutputInterface('Output');
    this.addOption(SparseAdamOptions.Lr, TypeOptions.SliderOption, 0.001);
    this.addOption(SparseAdamOptions.Betas, TypeOptions.VectorOption, [0.9, 0.999]);
    this.addOption(SparseAdamOptions.Eps, TypeOptions.SliderOption, 1e-08);
  }
}
