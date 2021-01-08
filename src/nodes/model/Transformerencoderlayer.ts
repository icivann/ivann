import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum TransformerEncoderLayerOptions {
  DModel = 'D model',
  Nhead = 'Nhead',
  DimFeedforward = 'Dim feedforward',
  Dropout = 'Dropout',
  Activation = 'Activation'
}
export default class TransformerEncoderLayer extends Node {
  type = ModelNodes.TransformerEncoderLayer;
  name = ModelNodes.TransformerEncoderLayer;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(TransformerEncoderLayerOptions.DModel, TypeOptions.IntOption, 0);
    this.addOption(TransformerEncoderLayerOptions.Nhead, TypeOptions.IntOption, 0);
    this.addOption(TransformerEncoderLayerOptions.DimFeedforward, TypeOptions.IntOption, 2048);
    this.addOption(TransformerEncoderLayerOptions.Dropout, TypeOptions.SliderOption, 0.1);
    this.addOption(TransformerEncoderLayerOptions.Activation, TypeOptions.DropdownOption, 'relu');
  }
}
