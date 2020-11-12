import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum SoftmaxOptions {
  Dim = 'Dim'
}
export default class Softmax extends Node {
  type = ModelNodes.Softmax;
  name = ModelNodes.Softmax;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(SoftmaxOptions.Dim, TypeOptions.VectorOption, [0]);
  }
}
