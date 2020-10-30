import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ConvTranspose1dOptions {
  In_channels = 'In channels',
  Out_channels = 'Out channels',
  Kernel_size = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Output_padding = 'Output padding',
  Groups = 'Groups',
  Bias = 'Bias',
  Dilation = 'Dilation',
  Padding_mode = 'Padding mode'
}
export default class ConvTranspose1d extends Node {
  type = Nodes.Convtranspose1d;
  name = Nodes.Convtranspose1d;

constructor() {
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  this.addOption(ConvTranspose1dOptions.In_channels, TypeOptions.IntOption, 0);
  this.addOption(ConvTranspose1dOptions.Out_channels, TypeOptions.IntOption, 0);
  this.addOption(ConvTranspose1dOptions.Kernel_size, TypeOptions.VectorOption, [0]);
  this.addOption(ConvTranspose1dOptions.Stride, TypeOptions.VectorOption, [1]);
  this.addOption(ConvTranspose1dOptions.Padding, TypeOptions.VectorOption, [0]);
  this.addOption(ConvTranspose1dOptions.Output_padding, TypeOptions.VectorOption, [0]);
  this.addOption(ConvTranspose1dOptions.Groups, TypeOptions.IntOption, 1);
  this.addOption(ConvTranspose1dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  this.addOption(ConvTranspose1dOptions.Dilation, TypeOptions.VectorOption, 1);
  this.addOption(ConvTranspose1dOptions.Padding_mode, TypeOptions.DropdownOption, 'zeros');
  }
}
