import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum FoldOptions {
  OutputSize = 'Output size',
  KernelSize = 'Kernel size',
  Dilation = 'Dilation',
  Padding = 'Padding',
  Stride = 'Stride'
}
export default class Fold extends Node {
  type = ModelNodes.Fold;
  name = ModelNodes.Fold;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(FoldOptions.OutputSize, TypeOptions.VectorOption, [0]);
    this.addOption(FoldOptions.KernelSize, TypeOptions.VectorOption, [0]);
    this.addOption(FoldOptions.Dilation, TypeOptions.VectorOption, [1]);
    this.addOption(FoldOptions.Padding, TypeOptions.VectorOption, [0]);
    this.addOption(FoldOptions.Stride, TypeOptions.VectorOption, [1]);
  }
}
