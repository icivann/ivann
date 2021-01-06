import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum RNNCellOptions {
  InputSize = 'Input size',
  HiddenSize = 'Hidden size',
  Bias = 'Bias',
  Nonlinearity = 'Nonlinearity'
}
export default class RNNCell extends Node {
  type = ModelNodes.RNNCell;
  name = ModelNodes.RNNCell;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(RNNCellOptions.InputSize, TypeOptions.IntOption, 0);
    this.addOption(RNNCellOptions.HiddenSize, TypeOptions.IntOption, 0);
    this.addOption(RNNCellOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(RNNCellOptions.Nonlinearity, TypeOptions.DropdownOption, 'tanh');
  }
}
