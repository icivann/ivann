import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum MaxUnpool2dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding'
}
export default class MaxUnpool2d extends Node {
  type = ModelNodes.MaxUnpool2d;
  name = ModelNodes.MaxUnpool2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MaxUnpool2dOptions.KernelSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(MaxUnpool2dOptions.Stride, TypeOptions.VectorOption, [0, 0]);
    this.addOption(MaxUnpool2dOptions.Padding, TypeOptions.VectorOption, [0, 0]);
  }
}
