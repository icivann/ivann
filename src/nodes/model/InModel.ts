import { Node } from '@baklavajs/core';
import { Nodes, Layers } from '@/nodes/model/Types';

export default class InModel extends Node {
  type = Layers.IO;
  name = Nodes.InModel;

  constructor() {
    super();

    this.addOutputInterface('Output');
  }
}
