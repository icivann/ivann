import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum GRUOptions {
  InputSize = 'Input size ',
  HiddenSize = 'Hidden size',
  NumLayers = 'Num layers',
  Bias = 'Bias',
  BatchFirst = 'Batch first',
  Dropout = 'Dropout',
  Bidirectional = 'Bidirectional'
}
export default class GRU extends Node {
  type = ModelNodes.GRU;
  name = ModelNodes.GRU;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(GRUOptions.InputSize, TypeOptions.IntOption, 0);
    this.addOption(GRUOptions.HiddenSize, TypeOptions.IntOption, 0);
    this.addOption(GRUOptions.NumLayers, TypeOptions.IntOption, 1);
    this.addOption(GRUOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(GRUOptions.BatchFirst, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(GRUOptions.Dropout, TypeOptions.SliderOption, 0.0);
    this.addOption(GRUOptions.Bidirectional, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
