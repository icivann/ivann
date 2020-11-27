import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LPPool1dOptions {
  NormType = 'Norm type',
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  CeilMode = 'Ceil mode'
}
export default class LPPool1d extends Node {
  type = ModelNodes.LPPool1d;
  name = ModelNodes.LPPool1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(LPPool1dOptions.NormType, TypeOptions.SliderOption, 0.0);
    this.addOption(LPPool1dOptions.KernelSize, TypeOptions.VectorOption, [0]);
    this.addOption(LPPool1dOptions.Stride, TypeOptions.VectorOption, [0]);
    this.addOption(LPPool1dOptions.CeilMode, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
