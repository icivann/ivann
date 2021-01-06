import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LPPool2dOptions {
  NormType = 'Norm type',
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  CeilMode = 'Ceil mode'
}
export default class LPPool2d extends Node {
  type = ModelNodes.LPPool2d;
  name = ModelNodes.LPPool2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(LPPool2dOptions.NormType, TypeOptions.SliderOption, [0, 0]);
    this.addOption(LPPool2dOptions.KernelSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(LPPool2dOptions.Stride, TypeOptions.VectorOption, [0, 0]);
    this.addOption(LPPool2dOptions.CeilMode, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
