import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum MaxPool2dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Dilation = 'Dilation',
  ReturnIndices = 'Return indices',
  CeilMode = 'Ceil mode'
}
export default class MaxPool2d extends Node {
  type = ModelNodes.MaxPool2d;
  name = ModelNodes.MaxPool2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MaxPool2dOptions.KernelSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(MaxPool2dOptions.Stride, TypeOptions.VectorOption, [0, 0]);
    this.addOption(MaxPool2dOptions.Padding, TypeOptions.VectorOption, [0, 0]);
    this.addOption(MaxPool2dOptions.Dilation, TypeOptions.VectorOption, [1, 1]);
    this.addOption(MaxPool2dOptions.ReturnIndices, TypeOptions.TickBoxOption);
    this.addOption(MaxPool2dOptions.CeilMode, TypeOptions.TickBoxOption);
  }
}
