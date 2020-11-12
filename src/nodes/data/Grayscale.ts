import { Node } from '@baklavajs/core';
import { Nodes, NodeTypes } from '@/nodes/data/Types';

export default class Grayscale extends Node {
  type = NodeTypes.Transform;
  name = Nodes.Grayscale;

  constructor() {
    super();

    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
