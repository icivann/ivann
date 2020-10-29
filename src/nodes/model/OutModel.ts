import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';

export default class OutModel extends Node {
  type = Layers.IO;
  name = Nodes.OutModel;

  constructor() {
    super();
    this.addInputInterface('Input');
  }
}
