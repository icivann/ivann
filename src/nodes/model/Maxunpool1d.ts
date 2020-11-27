import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum MaxUnpool1dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding'
}
export default class MaxUnpool1d extends Node {
  type = ModelNodes.MaxUnpool1d;
  name = ModelNodes.MaxUnpool1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MaxUnpool1dOptions.KernelSize, TypeOptions.VectorOption, [0]);
    this.addOption(MaxUnpool1dOptions.Stride, TypeOptions.VectorOption, [0]);
    this.addOption(MaxUnpool1dOptions.Padding, TypeOptions.VectorOption, [0]);
  }
}
