import { Node } from '@baklavajs/core';
import { OverviewNodes } from '@/nodes/overview/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum PoissonNLLLossOptions {
  LogInput = 'Log input',
  Full = 'Full',
  SizeAverage = 'Size average',
  Eps = 'Eps',
  Reduce = 'Reduce',
  Reduction = 'Reduction'
}
export default class PoissonNLLLoss extends Node {
  type = OverviewNodes.PoissonNLLLoss;
  name = OverviewNodes.PoissonNLLLoss;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(PoissonNLLLossOptions.LogInput, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(PoissonNLLLossOptions.Full, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(PoissonNLLLossOptions.SizeAverage, TypeOptions.IntOption, 0);
    this.addOption(PoissonNLLLossOptions.Eps, TypeOptions.SliderOption, 1e-08);
    this.addOption(PoissonNLLLossOptions.Reduce, TypeOptions.IntOption, 0);
    this.addOption(PoissonNLLLossOptions.Reduction, TypeOptions.DropdownOption, 'mean');
  }
}
