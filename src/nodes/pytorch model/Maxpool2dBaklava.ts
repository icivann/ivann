import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum MaxPool2dOptions {
  Kernel_size = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Dilation = 'Dilation',
  Return_indices = 'Return indices',
  Ceil_mode = 'Ceil mode'
}
export default class MaxPool2d extends Node {
  type = Nodes.MaxPool2D;
  name = Nodes.MaxPool2D;

constructor() {
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  this.addOption(MaxPool2dOptions.Kernel_size, TypeOptions.VectorOption, [0,0]);
  this.addOption(MaxPool2dOptions.Stride, TypeOptions.VectorOption, undefined);
  this.addOption(MaxPool2dOptions.Padding, TypeOptions.VectorOption, 0);
  this.addOption(MaxPool2dOptions.Dilation, TypeOptions.VectorOption, 1);
  this.addOption(MaxPool2dOptions.Return_indices, TypeOptions.TickBoxOption);
  this.addOption(MaxPool2dOptions.Ceil_mode, TypeOptions.TickBoxOption);
  }
}
