import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';

export default class Flatten extends Node {
  type = Nodes.Flatten;
  name = Nodes.Flatten;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
