import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum TransformerOptions {
  DModel = 'D model',
  Nhead = 'Nhead',
  NumEncoderLayers = 'Num encoder layers',
  NumDecoderLayers = 'Num decoder layers',
  DimFeedforward = 'Dim feedforward',
  Dropout = 'Dropout',
  Activation = 'Activation',
  CustomEncoder = 'Custom encoder',
  CustomDecoder = 'Custom decoder'
}
export default class Transformer extends Node {
  type = ModelNodes.Transformer;
  name = ModelNodes.Transformer;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(TransformerOptions.DModel, TypeOptions.IntOption, 512);
    this.addOption(TransformerOptions.Nhead, TypeOptions.IntOption, 8);
    this.addOption(TransformerOptions.NumEncoderLayers, TypeOptions.IntOption, 6);
    this.addOption(TransformerOptions.NumDecoderLayers, TypeOptions.IntOption, 6);
    this.addOption(TransformerOptions.DimFeedforward, TypeOptions.IntOption, 2048);
    this.addOption(TransformerOptions.Dropout, TypeOptions.SliderOption, 0.1);
    this.addOption(TransformerOptions.Activation, TypeOptions.DropdownOption, 'relu');
    this.addOption(TransformerOptions.CustomEncoder, TypeOptions.VectorOption, [0]);
    this.addOption(TransformerOptions.CustomDecoder, TypeOptions.VectorOption, [0]);
  }
}
