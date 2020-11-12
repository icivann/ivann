import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum SoftminOptions {
  Dim = 'Dim'
}
export default class Softmin extends Node {
  type = Nodes.Softmin;
  name = Nodes.Softmin;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(SoftminOptions.Dim, TypeOptions.VectorOption, [0]);
  }
}
