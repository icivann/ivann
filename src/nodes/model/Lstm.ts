import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LSTMOptions {
  InputSize = 'Input size ',
  HiddenSize = 'Hidden size',
  NumLayers = 'Num layers',
  Bias = 'Bias',
  BatchFirst = 'Batch first',
  Dropout = 'Dropout',
  Bidirectional = 'Bidirectional'
}
export default class LSTM extends Node {
  type = ModelNodes.LSTM;
  name = ModelNodes.LSTM;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(LSTMOptions.InputSize, TypeOptions.IntOption, 0);
    this.addOption(LSTMOptions.HiddenSize, TypeOptions.IntOption, 0);
    this.addOption(LSTMOptions.NumLayers, TypeOptions.IntOption, 1);
    this.addOption(LSTMOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(LSTMOptions.BatchFirst, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(LSTMOptions.Dropout, TypeOptions.SliderOption, 0.0);
    this.addOption(LSTMOptions.Bidirectional, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
