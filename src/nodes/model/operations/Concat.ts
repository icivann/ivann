import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/Types';

export default class Concat extends Node {
  type = Nodes.Concat;
  name = Nodes.Concat;

  constructor() {
    super();

    this.addInputInterface('Input 1');
    this.addInputInterface('Input 2');
    this.addOutputInterface('Output');
  }
}
