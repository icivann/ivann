import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum Conv3dOptions {
  In_channels = 'In channels',
  Out_channels = 'Out channels',
  Kernel_size = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Dilation = 'Dilation',
  Groups = 'Groups',
  Bias = 'Bias',
  Padding_mode = 'Padding mode'
}
export default class Conv3d extends Node {
  type = Nodes.Conv3D;
  name = Nodes.Conv3D;

constructor() {
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  this.addOption(Conv3dOptions.In_channels, TypeOptions.IntOption,0);
  this.addOption(Conv3dOptions.Out_channels, TypeOptions.IntOption,0);
  this.addOption(Conv3dOptions.Kernel_size, TypeOptions.VectorOption, [0,0,0]);
  this.addOption(Conv3dOptions.Stride, TypeOptions.VectorOption, [1,1,1]);
  this.addOption(Conv3dOptions.Padding, TypeOptions.VectorOption, [0,0,0]);
  this.addOption(Conv3dOptions.Dilation, TypeOptions.VectorOption, [1,1,1]);
  this.addOption(Conv3dOptions.Groups, TypeOptions.IntOption, 1);
  this.addOption(Conv3dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  this.addOption(Conv3dOptions.Padding_mode, TypeOptions.DropdownOption, 'zeros');
  }
}
