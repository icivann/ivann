import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum Conv1dOptions {
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

export default class Conv1d extends Node {
  type = Nodes.Conv1D;
  name = Nodes.Conv1D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(Conv1dOptions.In_channels, TypeOptions.IntOption, 0);
    this.addOption(Conv1dOptions.Out_channels, TypeOptions.IntOption, 0);
    this.addOption(Conv1dOptions.Kernel_size, TypeOptions.VectorOption, [0]);
    this.addOption(Conv1dOptions.Stride, TypeOptions.VectorOption, [1]);
    this.addOption(Conv1dOptions.Padding, TypeOptions.VectorOption, [0]);
    this.addOption(Conv1dOptions.Dilation, TypeOptions.VectorOption, [1]);
    this.addOption(Conv1dOptions.Groups, TypeOptions.IntOption, 1);
    this.addOption(Conv1dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(Conv1dOptions.Padding_mode, TypeOptions.DropdownOption, 'zeros');
  }
}
