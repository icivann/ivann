import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum AvgPool3dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  CeilMode = 'Ceil mode',
  CountIncludePad = 'Count include pad',
  DivisorOverride = 'Divisor override'
}
export default class AvgPool3d extends Node {
  type = ModelNodes.AvgPool3d;
  name = ModelNodes.AvgPool3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(AvgPool3dOptions.KernelSize, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(AvgPool3dOptions.Stride, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(AvgPool3dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(AvgPool3dOptions.CeilMode, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(AvgPool3dOptions.CountIncludePad, TypeOptions.TickBoxOption, CheckboxValue.CHECKED);
    this.addOption(AvgPool3dOptions.DivisorOverride, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
