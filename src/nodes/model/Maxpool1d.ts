import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import { Conv1dOptions } from '@/nodes/model/Conv1d';

export enum MaxPool1dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding',
  Dilation = 'Dilation',
  ReturnIndices = 'Return indices',
  CeilMode = 'Ceil mode'
}
export default class MaxPool1d extends Node {
  type = Nodes.MaxPool1d;
  name = Nodes.MaxPool1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MaxPool1dOptions.KernelSize, TypeOptions.VectorOption, [0]);
    this.addOption(MaxPool1dOptions.Stride, TypeOptions.VectorOption, [1]);
    this.addOption(MaxPool1dOptions.Padding, TypeOptions.VectorOption, [0]);
    this.addOption(MaxPool1dOptions.Dilation, TypeOptions.VectorOption, [1]);
    this.addOption(MaxPool1dOptions.ReturnIndices, TypeOptions.TickBoxOption);
    this.addOption(MaxPool1dOptions.CeilMode, TypeOptions.TickBoxOption);
  }
}
