import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum TransformerDecoderLayerOptions {
  DModel = 'D model',
  Nhead = 'Nhead',
  DimFeedforward = 'Dim feedforward',
  Dropout = 'Dropout',
  Activation = 'Activation'
}
export default class TransformerDecoderLayer extends Node {
  type = ModelNodes.TransformerDecoderLayer;
  name = ModelNodes.TransformerDecoderLayer;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(TransformerDecoderLayerOptions.DModel, TypeOptions.IntOption, 0);
    this.addOption(TransformerDecoderLayerOptions.Nhead, TypeOptions.IntOption, 0);
    this.addOption(TransformerDecoderLayerOptions.DimFeedforward, TypeOptions.IntOption, 2048);
    this.addOption(TransformerDecoderLayerOptions.Dropout, TypeOptions.SliderOption, 0.1);
    this.addOption(TransformerDecoderLayerOptions.Activation, TypeOptions.DropdownOption, 'relu');
  }
}
