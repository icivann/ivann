import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum GRUCellOptions {
  InputSize = 'Input size',
  HiddenSize = 'Hidden size',
  Bias = 'Bias'
}
export default class GRUCell extends Node {
  type = ModelNodes.GRUCell;
  name = ModelNodes.GRUCell;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(GRUCellOptions.InputSize, TypeOptions.IntOption, 0);
    this.addOption(GRUCellOptions.HiddenSize, TypeOptions.IntOption, 0);
    this.addOption(GRUCellOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
