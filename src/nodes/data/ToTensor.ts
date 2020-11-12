import { Node } from '@baklavajs/core';
import { Nodes, NodeTypes } from '@/nodes/data/Types';

export default class ToTensor extends Node {
  type = NodeTypes.Transform;
  name = Nodes.ToTensor;

  constructor() {
    super();

    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
