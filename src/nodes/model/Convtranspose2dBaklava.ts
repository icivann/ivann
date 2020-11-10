import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ConvTranspose2dOptions {
  InChannels = 'In channels',
  OutChannels = 'Out channels',
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  OutputPadding = 'Output padding',
  Groups = 'Groups',
  Bias = 'Bias',
  Dilation = 'Dilation',
  PaddingMode = 'Padding mode'
}
export default class ConvTranspose2d extends Node {
  type = Nodes.Convtranspose2d;
  name = Nodes.Convtranspose2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ConvTranspose2dOptions.InChannels, TypeOptions.IntOption, 0);
    this.addOption(ConvTranspose2dOptions.OutChannels, TypeOptions.IntOption, 0);
    this.addOption(ConvTranspose2dOptions.KernelSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(ConvTranspose2dOptions.Stride, TypeOptions.VectorOption, [1, 1]);
    this.addOption(ConvTranspose2dOptions.Padding, TypeOptions.VectorOption, [0, 0]);
    this.addOption(ConvTranspose2dOptions.OutputPadding, TypeOptions.VectorOption, [0, 0]);
    this.addOption(ConvTranspose2dOptions.Groups, TypeOptions.IntOption, 1);
    this.addOption(ConvTranspose2dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(ConvTranspose2dOptions.Dilation, TypeOptions.IntOption, 1);
    this.addOption(ConvTranspose2dOptions.PaddingMode, TypeOptions.DropdownOption, 'zeros');
  }
}
