import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { PaddingMode } from '@/app/ir/irCommon';

export enum ConvTranspose1dOptions {
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
export default class ConvTranspose1d extends Node {
  type = ModelNodes.ConvTranspose1d;
  name = ModelNodes.ConvTranspose1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ConvTranspose1dOptions.InChannels, TypeOptions.IntOption, 0);
    this.addOption(ConvTranspose1dOptions.OutChannels, TypeOptions.IntOption, 0);
    this.addOption(ConvTranspose1dOptions.KernelSize, TypeOptions.VectorOption, [0]);
    this.addOption(ConvTranspose1dOptions.Stride, TypeOptions.VectorOption, [1]);
    this.addOption(ConvTranspose1dOptions.Padding, TypeOptions.VectorOption, [0]);
    this.addOption(ConvTranspose1dOptions.OutputPadding, TypeOptions.VectorOption, [0]);
    this.addOption(ConvTranspose1dOptions.Groups, TypeOptions.IntOption, 1);
    this.addOption(ConvTranspose1dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(ConvTranspose1dOptions.Dilation, TypeOptions.VectorOption, 1);
    this.addOption(ConvTranspose1dOptions.PaddingMode, TypeOptions.DropdownOption, 'zeros',
      undefined, { items: valuesOf(PaddingMode) });
  }
}
