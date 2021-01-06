import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum RNNBaseOptions {
  Mode = 'Mode',
  InputSize = 'Input size',
  HiddenSize = 'Hidden size',
  NumLayers = 'Num layers',
  Bias = 'Bias',
  BatchFirst = 'Batch first',
  Dropout = 'Dropout',
  Bidirectional = 'Bidirectional'
}
export default class RNNBase extends Node {
  type = ModelNodes.RNNBase;
  name = ModelNodes.RNNBase;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(RNNBaseOptions.Mode, TypeOptions.DropdownOption, 0);
    this.addOption(RNNBaseOptions.InputSize, TypeOptions.IntOption, 0);
    this.addOption(RNNBaseOptions.HiddenSize, TypeOptions.IntOption, 0);
    this.addOption(RNNBaseOptions.NumLayers, TypeOptions.IntOption, 1);
    this.addOption(RNNBaseOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(RNNBaseOptions.BatchFirst, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(RNNBaseOptions.Dropout, TypeOptions.SliderOption, 0.0);
    this.addOption(RNNBaseOptions.Bidirectional, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
