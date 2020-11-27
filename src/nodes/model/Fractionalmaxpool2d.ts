import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum FractionalMaxPool2dOptions {
  KernelSize = 'Kernel size',
  OutputSize = 'Output size',
  OutputRatio = 'Output ratio',
  ReturnIndices = 'Return indices',
  RandomSamples = ' random samples'
}
export default class FractionalMaxPool2d extends Node {
  type = ModelNodes.FractionalMaxPool2d;
  name = ModelNodes.FractionalMaxPool2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(FractionalMaxPool2dOptions.KernelSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(FractionalMaxPool2dOptions.OutputSize, TypeOptions.VectorOption, [0, 0]);
    this.addOption(FractionalMaxPool2dOptions.OutputRatio, TypeOptions.VectorOption, [0, 0]);
    this.addOption(FractionalMaxPool2dOptions.ReturnIndices, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
    this.addOption(FractionalMaxPool2dOptions.RandomSamples, TypeOptions.TickBoxOption, CheckboxValue.UNCHECKED);
  }
}
