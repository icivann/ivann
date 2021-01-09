import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LBFGSOptions {
  Lr = 'Lr',
  MaxIter = 'Max iter',
  MaxEval = 'Max eval',
  ToleranceGrad = 'Tolerance grad',
  ToleranceChange = 'Tolerance change',
  HistorySize = 'History size'
}
export default class LBFGS extends Node {
  type = OverviewNodes.LBFGS;
  name = OverviewNodes.LBFGS;

  constructor() {
    super();
    this.addInputInterface('Model');
    this.addOutputInterface('Output');
    this.addOption(LBFGSOptions.Lr, TypeOptions.SliderOption, 1);
    this.addOption(LBFGSOptions.MaxIter, TypeOptions.IntOption, 20);
    this.addOption(LBFGSOptions.MaxEval, TypeOptions.IntOption, 0);
    this.addOption(LBFGSOptions.ToleranceGrad, TypeOptions.SliderOption, 1e-07);
    this.addOption(LBFGSOptions.ToleranceChange, TypeOptions.SliderOption, 1e-09);
    this.addOption(LBFGSOptions.HistorySize, TypeOptions.IntOption, 100);
  }
}
