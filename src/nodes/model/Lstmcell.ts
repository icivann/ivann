import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LSTMCellOptions {
  InputSize = 'Input size',
  HiddenSize = 'Hidden size',
  Bias = 'Bias'
}
export default class LSTMCell extends Node {
  type = ModelNodes.LSTMCell;
  name = ModelNodes.LSTMCell;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(LSTMCellOptions.InputSize, TypeOptions.IntOption, 0);
    this.addOption(LSTMCellOptions.HiddenSize, TypeOptions.IntOption, 0);
    this.addOption(LSTMCellOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
