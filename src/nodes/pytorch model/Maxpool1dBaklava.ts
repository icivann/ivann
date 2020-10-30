import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum MaxPool1dOptions {
  Kernel_size = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Dilation = 'Dilation',
  Return_indices = 'Return indices',
  Ceil_mode = 'Ceil mode'
}
export default class MaxPool1d extends Node {
  type = Nodes.MaxPool1D;
  name = Nodes.MaxPool1D;

constructor() {
super();
  this.addInputInterface('Input');
  this.addOutputInterface('Output');
  this.addOption(MaxPool1dOptions.Kernel_size, TypeOptions.VectorOption,[0]);
  this.addOption(MaxPool1dOptions.Stride, TypeOptions.VectorOption, undefined);
  this.addOption(MaxPool1dOptions.Padding, TypeOptions.VectorOption, 0);
  this.addOption(MaxPool1dOptions.Dilation, TypeOptions.VectorOption, 1);
  this.addOption(MaxPool1dOptions.Return_indices, TypeOptions.TickBoxOption);
  this.addOption(MaxPool1dOptions.Ceil_mode, TypeOptions.TickBoxOption);
  }
}
