import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ConvTranspose3dOptions {
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
export default class ConvTranspose3d extends Node {
  type = Nodes.Convtranspose3d;
  name = Nodes.Convtranspose3d;

constructor() {
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  this.addOption(ConvTranspose3dOptions.In_channels, TypeOptions.IntOption, 0);
  this.addOption(ConvTranspose3dOptions.Out_channels, TypeOptions.IntOption, 0);
  this.addOption(ConvTranspose3dOptions.Kernel_size, TypeOptions.VectorOption, [0,0,0]);
  this.addOption(ConvTranspose3dOptions.Stride, TypeOptions.VectorOption, [1,1,1]);
  this.addOption(ConvTranspose3dOptions.Padding, TypeOptions.VectorOption, [0,0,0]);
  this.addOption(ConvTranspose3dOptions.Output_padding, TypeOptions.VectorOption, [0,0,0]);
  this.addOption(ConvTranspose3dOptions.Groups, TypeOptions.IntOption, 1);
  this.addOption(ConvTranspose3dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  this.addOption(ConvTranspose3dOptions.Dilation, TypeOptions.VectorOption, 1);
  this.addOption(ConvTranspose3dOptions.Padding_mode, TypeOptions.DropdownOption, 'zeros');
  }
}
