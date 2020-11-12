import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';
import { valuesOf } from '@/app/util';
import { PaddingMode } from '@/app/ir/irCommon';

export enum Conv1dOptions {
  InChannels = 'In channels',
  OutChannels = 'Out channels',
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Dilation = 'Dilation',
  Groups = 'Groups',
  Bias = 'Bias',
  PaddingMode = 'Padding mode',
}

export default class Conv1d extends Node {
  type = ModelNodes.Conv1d;
  name = ModelNodes.Conv1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(Conv1dOptions.InChannels, TypeOptions.IntOption, 0);
    this.addOption(Conv1dOptions.OutChannels, TypeOptions.IntOption, 0);
    this.addOption(Conv1dOptions.KernelSize, TypeOptions.VectorOption, [0]);
    this.addOption(Conv1dOptions.Stride, TypeOptions.VectorOption, [1]);
    this.addOption(Conv1dOptions.Padding, TypeOptions.VectorOption, [0]);
    this.addOption(Conv1dOptions.Dilation, TypeOptions.VectorOption, [1]);
    this.addOption(Conv1dOptions.Groups, TypeOptions.IntOption, 1);
    this.addOption(Conv1dOptions.Bias, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(Conv1dOptions.PaddingMode, TypeOptions.DropdownOption, 'zeros',
      undefined, { items: valuesOf(PaddingMode) });
  }
}
