import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
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

    this.addOption(AdadeltaOptions.lr, TypeOptions.SliderOption, 1);
    this.addOption(AdadeltaOptions.rho, TypeOptions.SliderOption, 0.9);
    this.addOption(AdadeltaOptions.eps, TypeOptions.SliderOption, 1e-06);
    this.addOption(AdadeltaOptions.weightDecay, TypeOptions.SliderOption, 0);

    // this.addInputInterface(AdadeltaOptions.Params);
    this.addOutputInterface('output');
  }

  public initCode(params: string[]): string {
    return `torch.optim.Adadelta(${params[0]}.parameters(), lr=1, rho=0.9, eps=0.000001, weight_decay=0)`;
  }
}
