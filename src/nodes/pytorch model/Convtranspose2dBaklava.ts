import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ConvTranspose2dOptions {
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
export default class ConvTranspose2d extends Node {
  type = Nodes.Convtranspose2d;
  name = Nodes.Convtranspose2d;

constructor() {
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  this.addOption(ConvTranspose2dOptions.In_channels, TypeOptions.IntOption, 0);
  this.addOption(ConvTranspose2dOptions.Out_channels, TypeOptions.IntOption, 0);
  this.addOption(ConvTranspose2dOptions.Kernel_size, TypeOptions.VectorOption, [0,0]);
  this.addOption(ConvTranspose2dOptions.Stride, TypeOptions.VectorOption, [1,1]);
  this.addOption(ConvTranspose2dOptions.Padding, TypeOptions.VectorOption, [0,0]);
  this.addOption(ConvTranspose2dOptions.Output_padding, TypeOptions.VectorOption, [0,0]);
  this.addOption(ConvTranspose2dOptions.Groups, TypeOptions.IntOption, 1);
  this.addOption(ConvTranspose2dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  this.addOption(ConvTranspose2dOptions.Dilation, TypeOptions.IntOption, 1);
  this.addOption(ConvTranspose2dOptions.Padding_mode, TypeOptions.DropdownOption, 'zeros');
  }
}
