import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum SoftmaxOptions {
  Dim = 'Dim'
}
export default class Softmax extends Node {
  type = Nodes.Softmax;
  name = Nodes.Softmax;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(SoftmaxOptions.Dim, TypeOptions.VectorOption, undefined);
  }
}
