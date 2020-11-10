import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/Types';

export default class OutModel extends Node {
  type = Nodes.OutModel;
  name = Nodes.OutModel;

  constructor() {
    super();
    this.addInputInterface('Input');
  }
}
