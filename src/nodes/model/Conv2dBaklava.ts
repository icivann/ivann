import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum Conv2dOptions {
  InChannels = 'In channels',
  OutChannels = 'Out channels',
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Dilation = 'Dilation',
  Groups = 'Groups',
  Bias = 'Bias',
  PaddingMode = 'Padding mode'
}
export default class Conv2d extends Node {
  type = Nodes.Conv2D;
  name = Nodes.Conv2D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(Conv2dOptions.InChannels, TypeOptions.IntOption, 0);
    this.addOption(Conv2dOptions.OutChannels, TypeOptions.IntOption, 0);
    this.addOption(Conv2dOptions.KernelSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(Conv2dOptions.Stride, TypeOptions.VectorOption, [1, 1]);
    this.addOption(Conv2dOptions.Padding, TypeOptions.VectorOption, [0, 0]);
    this.addOption(Conv2dOptions.Dilation, TypeOptions.VectorOption, [1, 1]);
    this.addOption(Conv2dOptions.Groups, TypeOptions.IntOption, 1);
    this.addOption(Conv2dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(Conv2dOptions.PaddingMode, TypeOptions.DropdownOption, 'zeros');
  }
}
