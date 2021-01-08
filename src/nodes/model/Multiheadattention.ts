import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum MultiheadAttentionOptions {
  EmbedDim = 'Embed dim',
  NumHeads = 'Num heads',
  Dropout = 'Dropout',
  Bias = 'Bias',
  AddBiasKv = 'Add bias kv',
  AddZeroAttn = 'Add zero attn',
  Kdim = 'Kdim',
  Vdim = 'Vdim'
}
export default class MultiheadAttention extends Node {
  type = ModelNodes.MultiheadAttention;
  name = ModelNodes.MultiheadAttention;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MultiheadAttentionOptions.EmbedDim, TypeOptions.IntOption, 0);
    this.addOption(MultiheadAttentionOptions.NumHeads, TypeOptions.IntOption, 0);
    this.addOption(MultiheadAttentionOptions.Dropout, TypeOptions.SliderOption, 0.0);
    this.addOption(MultiheadAttentionOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(MultiheadAttentionOptions.AddBiasKv, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(MultiheadAttentionOptions.AddZeroAttn, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(MultiheadAttentionOptions.Kdim, TypeOptions.IntOption, 0);
    this.addOption(MultiheadAttentionOptions.Vdim, TypeOptions.IntOption, 0);
  }
}
