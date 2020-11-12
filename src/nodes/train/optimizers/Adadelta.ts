import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum AdadeltaOptions {
  Params = 'params',
  lr = 'lr',
  rho = 'rho',
  eps = 'eps',
  weightDecay = 'weightDecay',
}

export default class Adadelta extends Node {
  type = OverviewNodes.Adadelta;
  name = OverviewNodes.Adadelta;

  constructor() {
    super();

    this.addOption(AdadeltaOptions.lr, TypeOptions.SliderOption);
    this.addOption(AdadeltaOptions.rho, TypeOptions.SliderOption);
    this.addOption(AdadeltaOptions.eps, TypeOptions.SliderOption);
    this.addOption(AdadeltaOptions.weightDecay, TypeOptions.SliderOption);

    this.addInputInterface(AdadeltaOptions.Params);
    this.addOutputInterface('output');
  }
}
