import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AvgPool1dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  CeilMode = 'Ceil mode',
  CountIncludePad = 'Count include pad'
}
export default class AvgPool1d extends Node {
  type = ModelNodes.AvgPool1d;
  name = ModelNodes.AvgPool1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AvgPool1dOptions.KernelSize, TypeOptions.VectorOption, [0]);
    this.addOption(AvgPool1dOptions.Stride, TypeOptions.VectorOption, [0]);
    this.addOption(AvgPool1dOptions.Padding, TypeOptions.VectorOption, [0]);
    this.addOption(AvgPool1dOptions.CeilMode, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(AvgPool1dOptions.CountIncludePad, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
  }
}
