import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum MaxPool3dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Dilation = 'Dilation',
  ReturnIndices = 'Return indices',
  CeilMode = 'Ceil mode'
}
export default class MaxPool3d extends Node {
  type = ModelNodes.MaxPool3d;
  name = ModelNodes.MaxPool3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MaxPool3dOptions.KernelSize, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(MaxPool3dOptions.Stride, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(MaxPool3dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(MaxPool3dOptions.Dilation, TypeOptions.VectorOption, [1, 1, 1]);
    this.addOption(MaxPool3dOptions.ReturnIndices, TypeOptions.TickBoxOption);
    this.addOption(MaxPool3dOptions.CeilMode, TypeOptions.TickBoxOption);
  }
}
