import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum RNNOptions {
  InputSize = 'Input size ',
  HiddenSize = 'Hidden size',
  NumLayers = 'Num layers',
  Nonlinearity = 'Nonlinearity',
  Bias = 'Bias',
  BatchFirst = 'Batch first',
  Dropout = 'Dropout',
  Bidirectional = 'Bidirectional'
}
export default class RNN extends Node {
  type = ModelNodes.RNN;
  name = ModelNodes.RNN;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(RNNOptions.InputSize, TypeOptions.IntOption, 0);
    this.addOption(RNNOptions.HiddenSize, TypeOptions.IntOption, 0);
    this.addOption(RNNOptions.NumLayers, TypeOptions.IntOption, 1);
    this.addOption(RNNOptions.Nonlinearity, TypeOptions.DropdownOption, 'tanh');
    this.addOption(RNNOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(RNNOptions.BatchFirst, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(RNNOptions.Dropout, TypeOptions.SliderOption, 0.0);
    this.addOption(RNNOptions.Bidirectional, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
