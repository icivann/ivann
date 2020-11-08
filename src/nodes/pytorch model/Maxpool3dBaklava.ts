import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
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
  type = Nodes.MaxPool3D;
  name = Nodes.MaxPool3D;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MaxPool3dOptions.KernelSize, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(MaxPool3dOptions.Stride, TypeOptions.VectorOption, undefined);
    this.addOption(MaxPool3dOptions.Padding, TypeOptions.VectorOption, 0);
    this.addOption(MaxPool3dOptions.Dilation, TypeOptions.VectorOption, 1);
    this.addOption(MaxPool3dOptions.ReturnIndices, TypeOptions.TickBoxOption);
    this.addOption(MaxPool3dOptions.CeilMode, TypeOptions.TickBoxOption);
  }
}
