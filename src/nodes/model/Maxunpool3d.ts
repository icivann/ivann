import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum MaxUnpool3dOptions {
  KernelSize = 'Kernel size',
  Stride = 'Stride',
  Padding = 'Padding'
}
export default class MaxUnpool3d extends Node {
  type = ModelNodes.MaxUnpool3d;
  name = ModelNodes.MaxUnpool3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(MaxUnpool3dOptions.KernelSize, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(MaxUnpool3dOptions.Stride, TypeOptions.VectorOption, [0, 0, 0]);
    this.addOption(MaxUnpool3dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0]);
  }
}
