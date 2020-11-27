import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AvgPool2dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  CeilMode = 'Ceil mode',
  CountIncludePad = 'Count include pad',
  DivisorOverride = 'Divisor override'
}
export default class AvgPool2d extends Node {
  type = ModelNodes.AvgPool2d;
  name = ModelNodes.AvgPool2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AvgPool2dOptions.KernelSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(AvgPool2dOptions.Stride, TypeOptions.VectorOption, [0, 0]);
    this.addOption(AvgPool2dOptions.Padding, TypeOptions.VectorOption, [0, 0]);
    this.addOption(AvgPool2dOptions.CeilMode, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(AvgPool2dOptions.CountIncludePad, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(AvgPool2dOptions.DivisorOverride, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
