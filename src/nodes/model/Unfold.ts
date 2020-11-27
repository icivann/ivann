import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum UnfoldOptions {
  KernelSize = 'Kernel size',
  Dilation = 'Dilation',
  Padding = 'Padding',
  Stride = 'Stride'
}
export default class Unfold extends Node {
  type = ModelNodes.Unfold;
  name = ModelNodes.Unfold;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(UnfoldOptions.KernelSize, TypeOptions.VectorOption, [0]);
    this.addOption(UnfoldOptions.Dilation, TypeOptions.VectorOption, [1]);
    this.addOption(UnfoldOptions.Padding, TypeOptions.VectorOption, [0]);
    this.addOption(UnfoldOptions.Stride, TypeOptions.VectorOption, [1]);
  }
}
