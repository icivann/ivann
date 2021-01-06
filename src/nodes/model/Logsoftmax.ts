import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum LogSoftmaxOptions {
  Dim = 'Dim'
}
export default class LogSoftmax extends Node {
  type = ModelNodes.LogSoftmax;
  name = ModelNodes.LogSoftmax;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(LogSoftmaxOptions.Dim, TypeOptions.VectorOption, 0);
  }
}
